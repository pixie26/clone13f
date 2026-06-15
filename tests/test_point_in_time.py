import pandas as pd

from engine import PortfolioConfig, manager_characteristics, target_weights


def _row(manager, period, filing, accession, ticker, value=100.0):
    return {
        "manager": manager,
        "period_date": pd.Timestamp(period),
        "filing_date": pd.Timestamp(filing),
        "accession_number": accession,
        "submission_type": "13F-HR",
        "ticker": ticker,
        "value": value,
        "sec_type": "SH",
    }


def test_amendment_versions_are_not_blended():
    holdings = pd.DataFrame(
        [
            _row("m1", "2020-03-31", "2020-05-15", "orig", "A"),
            _row("m1", "2020-03-31", "2020-06-15", "amend", "B"),
        ]
    )

    chars = manager_characteristics(holdings)

    assert len(chars) == 2
    books = {r.accession_number: set(r.bw.index) for r in chars.itertuples()}
    assert books == {"orig": {"A"}, "amend": {"B"}}


def test_amendment_is_visible_only_after_its_filing_date():
    holdings = pd.DataFrame(
        [
            _row("m1", "2020-03-31", "2020-05-15", "orig", "A"),
            _row("m1", "2020-03-31", "2020-06-15", "amend", "B"),
        ]
    )
    chars = manager_characteristics(holdings)
    cfg = PortfolioConfig(top_n_ideas=1, consensus_weight=True)

    before = target_weights(chars, ["m1"], pd.Timestamp("2020-05-31"), cfg)
    after = target_weights(chars, ["m1"], pd.Timestamp("2020-06-30"), cfg)

    assert before.index.tolist() == ["A"]
    assert after.index.tolist() == ["B"]


def test_late_amendment_to_old_period_does_not_override_newer_report_period():
    holdings = pd.DataFrame(
        [
            _row("m1", "2020-03-31", "2020-05-15", "q1-orig", "A"),
            _row("m1", "2020-06-30", "2020-08-14", "q2-orig", "C"),
            _row("m1", "2020-03-31", "2020-09-01", "q1-amend", "B"),
        ]
    )
    chars = manager_characteristics(holdings)
    cfg = PortfolioConfig(top_n_ideas=1, consensus_weight=True)

    weights = target_weights(chars, ["m1"], pd.Timestamp("2020-09-15"), cfg)

    assert weights.index.tolist() == ["C"]
