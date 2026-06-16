import io
import zipfile

import pandas as pd

from build_universe import parse_quarter


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
