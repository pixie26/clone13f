import io
import zipfile

import pandas as pd

import build_universe as bu
from build_universe import coarse_prefilter, parse_quarter


def _zip_with_tsvs(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def test_parse_quarter_preserves_prn_security_type():
    payload = _zip_with_tsvs(
        {
            "SUBMISSION.TSV": (
                "ACCESSION_NUMBER\tSUBMISSIONTYPE\tFILING_DATE\tPERIODOFREPORT\tCIK\n"
                "a1\t13F-HR\t31-MAR-2026\t31-MAR-2026\t123\n"
            ),
            "COVERPAGE.TSV": "ACCESSION_NUMBER\tFILINGMANAGER_NAME\n" "a1\tTest Manager\n",
            "INFOTABLE.TSV": (
                "ACCESSION_NUMBER\tCUSIP\tNAMEOFISSUER\tVALUE\tSSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\n"
                "a1\t111111111\tCommon Co\t100\t10\tSH\t\n"
                "a1\t222222222\tConvertible Co\t200\t20\tPRN\t\n"
                "a1\t333333333\tPut Co\t300\t30\tSH\tPUT\n"
            ),
        }
    )

    parsed = parse_quarter(payload).sort_values("cusip")

    assert parsed[["cusip", "sec_type"]].to_dict("records") == [
        {"cusip": "111111111", "sec_type": "SH"},
        {"cusip": "222222222", "sec_type": "PRN"},
        {"cusip": "333333333", "sec_type": "PUT"},
    ]
    assert parsed.loc[parsed["cusip"].eq("222222222"), "share_amount_type"].iat[0] == "PRN"


def test_coarse_prefilter_screens_latest_accession_not_sum_of_amendments():
    holdings = pd.DataFrame(
        [
            {
                "cik": "1",
                "period_date": pd.Timestamp("2025-03-31"),
                "filing_date": pd.Timestamp("2025-05-01"),
                "accession_number": "orig",
                "cusip": "111111111",
                "value": 20e9,
                "sec_type": "SH",
            },
            {
                "cik": "1",
                "period_date": pd.Timestamp("2025-03-31"),
                "filing_date": pd.Timestamp("2025-05-15"),
                "accession_number": "amend",
                "cusip": "111111111",
                "value": 20e9,
                "sec_type": "SH",
            },
        ]
    )

    kept = coarse_prefilter(holdings, min_aum=1e9, max_aum=30e9, max_holdings=60)

    assert kept["accession_number"].tolist() == ["orig", "amend"]


def test_build_holdings_universe_uses_processed_universe_cache(monkeypatch, tmp_path):
    payload = _zip_with_tsvs(
        {
            "SUBMISSION.TSV": (
                "ACCESSION_NUMBER\tSUBMISSIONTYPE\tFILING_DATE\tPERIODOFREPORT\tCIK\n"
                "a1\t13F-HR\t15-MAY-2025\t31-MAR-2025\t123\n"
            ),
            "COVERPAGE.TSV": "ACCESSION_NUMBER\tFILINGMANAGER_NAME\n" "a1\tTest Manager\n",
            "INFOTABLE.TSV": (
                "ACCESSION_NUMBER\tCUSIP\tNAMEOFISSUER\tVALUE\tSSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\n"
                "a1\t111111111\tCommon Co\t2000000000\t10\tSH\t\n"
            ),
        }
    )
    url = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2025q1_form13f.zip"
    monkeypatch.setattr(bu, "discover_dataset_urls", lambda *args, **kwargs: [bu.DatasetURL(url)])
    monkeypatch.setattr(bu, "_try_download", lambda *args, **kwargs: payload)

    first = bu.build_holdings_universe(
        "2025-01-01",
        "2025-03-31",
        "Test test@example.com",
        cache_dir=str(tmp_path),
        min_aum=1e9,
        max_aum=30e9,
        max_holdings=60,
        max_put_weight=0.10,
    )
    for path in tmp_path.glob("2025q1*.parquet"):
        path.unlink()

    def fail_download(*args, **kwargs):
        raise AssertionError("processed universe cache should avoid raw archive download")

    monkeypatch.setattr(bu, "_try_download", fail_download)
    second = bu.build_holdings_universe(
        "2025-01-01",
        "2025-03-31",
        "Test test@example.com",
        cache_dir=str(tmp_path),
        min_aum=1e9,
        max_aum=30e9,
        max_holdings=60,
        max_put_weight=0.10,
    )

    assert len(first) == 1
    assert second[["accession_number", "cusip", "value"]].to_dict("records") == first[
        ["accession_number", "cusip", "value"]
    ].to_dict("records")
    assert list(tmp_path.glob("processed_universe.*.parquet"))
