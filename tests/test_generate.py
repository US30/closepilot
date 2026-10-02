import pandas as pd
import pytest

from data.generate import generate


@pytest.fixture(scope="module")
def t(tmp_path_factory):
    return generate(seed=7, out=tmp_path_factory.mktemp("raw"))


def test_referential_integrity(t):
    assert set(t["payments"].invoice_id) <= set(t["invoices"].invoice_id)
    assert set(t["recon_truth"].gl_id) == set(t["gl"].gl_id)
    assert set(t["labels"][t["labels"].entity == "invoice"].entity_id) <= set(t["invoices"].invoice_id)


def test_gst_math_clean_invoices(t):
    inv = t["invoices"]
    bad = set(t["labels"][t["labels"].anomaly_type == "gst_error"].entity_id)
    clean = inv[~inv.invoice_id.isin(bad)]
    tax = clean.cgst + clean.sgst + clean.igst
    assert ((clean.base_amount * clean.gst_rate - tax).abs() < 0.05).all()


def test_anomaly_counts(t):
    c = t["labels"].anomaly_type.value_counts()
    assert c["duplicate_invoice"] == 18 and c["split_invoice"] == 16 and c["bank_change"] == 4


def test_deterministic(tmp_path):
    a = generate(seed=3, out=tmp_path / "a")["gl"]
    b = generate(seed=3, out=tmp_path / "b")["gl"]
    pd.testing.assert_frame_equal(a, b)
