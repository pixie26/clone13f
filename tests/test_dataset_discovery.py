import pandas as pd

import build_universe as bu


def test_infer_window_from_legacy_quarter_url():
    start, end = bu._infer_window_from_url(
        "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2023q4_form13f.zip"
    )

    assert start == pd.Timestamp("2023-10-01")
    assert end == pd.Timestamp("2023-12-31")


def test_discovery_filters_scraped_legacy_urls(monkeypatch):
    class Response:
        text = """
        <a href="/files/structureddata/data/form-13f-data-sets/2023q4_form13f.zip">2023Q4</a>
        <a href="/files/structureddata/data/form-13f-data-sets/01jan2025-28feb2025_form13f.zip">2025 Jan-Feb</a>
        """

        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        return Response()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    urls = bu.discover_dataset_urls(
        "Test test@example.com",
        filing_start="2024-12-22",
        filing_end="2026-06-14",
    )
    names = [u.url.rsplit("/", 1)[-1] for u in urls]

    assert "2023q4_form13f.zip" not in names
    assert "01jan2025-28feb2025_form13f.zip" in names


def test_discovery_dedupes_base_and_dera_legacy_fallback(monkeypatch):
    class Response:
        text = """
        <a href="/files/structureddata/data/form-13f-data-sets/2013q4_form13f.zip">2013Q4</a>
        """

        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        return Response()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    urls = bu.discover_dataset_urls(
        "Test test@example.com",
        filing_start="2013-10-01",
        filing_end="2013-12-31",
    )
    names = [u.url.rsplit("/", 1)[-1] for u in urls]

    assert names == ["2013q4_form13f.zip"]
    assert "dera/data/form-13f" not in urls[0].url


def test_build_universe_filing_window_starts_after_report_period(monkeypatch):
    captured = {}

    def fake_discover(identity, filing_start, filing_end):
        captured["filing_start"] = pd.Timestamp(filing_start)
        return []

    monkeypatch.setattr(bu, "discover_dataset_urls", fake_discover)

    try:
        bu.build_holdings_universe(
            "2013-10-01",
            "2014-03-31",
            "Test test@example.com",
            cache_dir=None,
        )
    except RuntimeError:
        pass

    assert captured["filing_start"] == pd.Timestamp("2014-01-01")
