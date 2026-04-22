"""
Credit ledger — SQLite-backed, coordinator-signed receipt store.
Credits are denominated in inference tokens (IT).
1 IT = 1 token processed at full (FP16) precision.
INT4 earns 0.9 IT/token; 1-bit earns 0.5 IT/token.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

QUANT_RATES: dict[str, float] = {
    "fp16": 1.0,
    "int8": 0.95,
    "int4": 0.9,
    "awq-int4": 0.9,
    "gptq-int4": 0.9,
    "1bit": 0.5,
    "bitnet": 0.5,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    job_id          TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    node_id         TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    shard_range_lo  INTEGER NOT NULL,
    shard_range_hi  INTEGER NOT NULL,
    tokens_processed INTEGER NOT NULL,
    credits_earned  REAL NOT NULL,
    credits_spent   REAL NOT NULL DEFAULT 0,
    quantization    TEXT NOT NULL DEFAULT 'int4',
    coordinator_sig TEXT NOT NULL DEFAULT '',
    toploc_proof    TEXT NOT NULL DEFAULT '',
    direction       TEXT NOT NULL DEFAULT 'earned'
);

CREATE TABLE IF NOT EXISTS balance_cache (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    balance REAL NOT NULL DEFAULT 0
);

INSERT OR IGNORE INTO balance_cache (id, balance) VALUES (1, 0);
"""


@dataclass
class Receipt:
    job_id: str
    timestamp: float
    node_id: str
    model_id: str
    shard_range: tuple[int, int]
    tokens_processed: int
    credits_earned: float
    credits_spent: float = 0.0
    quantization: str = "int4"
    coordinator_sig: str = ""
    toploc_proof: str = ""
    direction: str = "earned"   # "earned" | "spent"

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "model": self.model_id,
            "shard_range": list(self.shard_range),
            "tokens_processed": self.tokens_processed,
            "credits_earned": self.credits_earned,
            "coordinator_sig": self.coordinator_sig,
            "toploc_proof": self.toploc_proof,
        }


class CreditLedger:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        self._con.executescript(SCHEMA)
        self._con.commit()

    @contextmanager
    def _tx(self):
        cur = self._con.cursor()
        try:
            yield cur
            self._con.commit()
        except Exception:
            self._con.rollback()
            raise

    def record_earned(
        self,
        job_id: str,
        node_id: str,
        model_id: str,
        shard_range: tuple[int, int],
        tokens: int,
        quantization: str = "int4",
        coordinator_sig: str = "",
        toploc_proof: str = "",
    ) -> Receipt:
        rate = QUANT_RATES.get(quantization.lower(), 0.9)
        earned = round(tokens * rate, 4)
        receipt = Receipt(
            job_id=job_id,
            timestamp=time.time(),
            node_id=node_id,
            model_id=model_id,
            shard_range=shard_range,
            tokens_processed=tokens,
            credits_earned=earned,
            quantization=quantization,
            coordinator_sig=coordinator_sig,
            toploc_proof=toploc_proof,
            direction="earned",
        )
        with self._tx() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO receipts
                   (job_id, timestamp, node_id, model_id, shard_range_lo, shard_range_hi,
                    tokens_processed, credits_earned, quantization, coordinator_sig,
                    toploc_proof, direction)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    receipt.job_id, receipt.timestamp, receipt.node_id,
                    receipt.model_id, shard_range[0], shard_range[1],
                    receipt.tokens_processed, earned,
                    quantization, coordinator_sig, toploc_proof, "earned",
                ),
            )
            cur.execute("UPDATE balance_cache SET balance = balance + ? WHERE id = 1", (earned,))
        return receipt

    def record_spent(self, job_id: str, node_id: str, model_id: str, tokens: int,
                     model_cost_factor: float = 1.0, priority_mul: float = 1.0) -> float:
        spent = round(tokens * model_cost_factor * priority_mul, 4)
        with self._tx() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO receipts
                   (job_id, timestamp, node_id, model_id, shard_range_lo, shard_range_hi,
                    tokens_processed, credits_earned, credits_spent, direction)
                   VALUES (?,?,?,?,0,0,?,0,?,?)""",
                (job_id, time.time(), node_id, model_id, tokens, spent, "spent"),
            )
            cur.execute("UPDATE balance_cache SET balance = balance - ? WHERE id = 1", (spent,))
        return spent

    def balance(self) -> float:
        row = self._con.execute("SELECT balance FROM balance_cache WHERE id = 1").fetchone()
        return row[0] if row else 0.0

    def recent_receipts(self, limit: int = 50) -> list[Receipt]:
        rows = self._con.execute(
            "SELECT * FROM receipts ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            out.append(Receipt(
                job_id=r[0], timestamp=r[1], node_id=r[2], model_id=r[3],
                shard_range=(r[4], r[5]), tokens_processed=r[6],
                credits_earned=r[7], credits_spent=r[8],
                quantization=r[9], coordinator_sig=r[10],
                toploc_proof=r[11], direction=r[12],
            ))
        return out

    def earn_spend_ratio(self, window_hours: float = 24.0) -> float:
        since = time.time() - window_hours * 3600
        row = self._con.execute(
            """SELECT
               SUM(CASE WHEN direction='earned' THEN credits_earned ELSE 0 END),
               SUM(CASE WHEN direction='spent'  THEN credits_spent  ELSE 0 END)
               FROM receipts WHERE timestamp > ?""",
            (since,),
        ).fetchone()
        earned, spent = (row[0] or 0.0), (row[1] or 0.0)
        return earned / spent if spent > 0 else float("inf")

    def close(self) -> None:
        self._con.close()
