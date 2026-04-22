"""Unit tests for the credit ledger."""
import pytest
from pathlib import Path
from opengrid.daemon.credit_ledger import CreditLedger


@pytest.fixture
def ledger(tmp_path):
    return CreditLedger(tmp_path / "ledger.db")


def test_initial_balance_zero(ledger):
    assert ledger.balance() == 0.0


def test_earn_credits(ledger):
    ledger.record_earned("job-1", "node-x", "llama3-8b-int4", (0, 7), tokens=512, quantization="int4")
    assert ledger.balance() == pytest.approx(512 * 0.9, rel=1e-4)


def test_spend_credits(ledger):
    ledger.record_earned("job-1", "node-x", "llama3-8b-int4", (0, 7), tokens=10000)
    ledger.record_spent("job-2", "node-x", "llama3-8b-int4", tokens=512, model_cost_factor=1.0)
    assert ledger.balance() > 0


def test_earn_spend_ratio(ledger):
    ledger.record_earned("job-1", "node-x", "llama3-8b-int4", (0, 7), tokens=10000)
    ratio = ledger.earn_spend_ratio()
    assert ratio == float("inf")  # no spending yet
