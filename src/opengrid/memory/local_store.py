"""
Local vector store — wraps LanceDB for personal long-term memory.
Data never leaves the user's machine.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class LocalMemoryStore:
    def __init__(self, store_dir: Path):
        self._dir = store_dir
        self._db = None
        self._table = None
        store_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        try:
            import lancedb  # type: ignore
            self._db = lancedb.connect(str(self._dir))
            log.info("LanceDB connected at %s", self._dir)
        except ImportError:
            log.warning("lancedb not installed — memory store disabled. Run: pip install lancedb")

    def add(self, text: str, metadata: dict | None = None, embedding: list[float] | None = None) -> None:
        if self._db is None:
            return
        try:
            import pyarrow as pa  # type: ignore
            record = {"text": text, "metadata": json.dumps(metadata or {})}
            if embedding:
                record["vector"] = embedding
            tbl_name = "memory"
            if tbl_name not in self._db.table_names():
                schema = pa.schema([
                    pa.field("text", pa.string()),
                    pa.field("metadata", pa.string()),
                ])
                if embedding:
                    schema = schema.append(pa.field("vector", pa.list_(pa.float32(), len(embedding))))
                self._table = self._db.create_table(tbl_name, schema=schema)
            else:
                self._table = self._db.open_table(tbl_name)
            self._table.add([record])
        except Exception as e:
            log.warning("LocalMemoryStore.add failed: %s", e)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if self._db is None or self._table is None:
            return []
        try:
            results = self._table.search(query_embedding).limit(top_k).to_list()
            return results
        except Exception as e:
            log.warning("LocalMemoryStore.search failed: %s", e)
            return []
