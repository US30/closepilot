import pytest

from closepilot.agents import copilot
from eval.gold_sql import GOLD


@pytest.mark.parametrize("sql", [
    "DROP TABLE invoices", "SELECT * FROM labels", "SELECT * FROM recon_truth", "SELECT 1; DELETE FROM gl",
    "SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM audit_log", "COPY invoices TO 'x.csv'",
])
def test_guard_blocks(sql):
    with pytest.raises(ValueError):
        copilot.guard(sql)


def test_guard_allows_and_limits():
    assert copilot.guard("SELECT count(*) FROM invoices").startswith("SELECT * FROM (")
    assert copilot.guard("select 1 limit 5").endswith("limit 5")


def test_gold_reference_sql_passes_guard():
    for _, sql in GOLD:
        copilot.guard(sql)
