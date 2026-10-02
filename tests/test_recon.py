import pytest

from closepilot.agents import recon


def test_recon_quality(con):
    res = recon.run_recon(con, use_llm=False)
    score = recon.score_against_truth(con, res["matches"])
    assert score["precision"] > 0.97
    assert score["recall"] > 0.9
    assert res["stats"].auto_match_rate[0] > 0.9


def test_bank_charges_become_exceptions(con):
    res = recon.run_recon(con, use_llm=False)
    assert (res["exceptions"].category == "bank_charge").sum() >= 10
