"""
Rule-based SEC 13F universe builder.

Key fixes vs the initial version:
  - SEC changed the bulk-file naming convention from simple YYYYqN archives to
    filing-month windows starting in 2024, e.g. 01mar2024-31may2024_form13f.zip.
  - start/end are treated as REPORT-PERIOD bounds; we download all filing-window
    archives that can contain those report periods, then filter period_date.
  - Dates are parsed with explicit SEC formats, avoiding pandas dateutil warnings.
  - Amendments are de-duplicated: latest filing per (CIK, report period) wins.
  - Stable manager key is CIK; manager_name is kept separately for display.
"""
from __future__ import annotations

import calendar
import io
import os
import re
import time
import zipfile
from dataclasses import dataclass
from urllib.parse import urljoin

import numpy as np
import pandas as pd

SEC_13F_BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
SEC_13F_LANDING = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
PARSED_CACHE_VERSION = "v2"

# Fallback only. The downloader first tries to scrape the official SEC landing page.
LEGACY_URL_PATTERN = SEC_13F_BASE + "{y}q{q}_form13f.zip"
OLDER_DERA_PATTERN = "https://www.sec.gov/files/dera/data/form-13f/{y}q{q}_form13f.zip"


@dataclass(frozen=True)
class DatasetURL:
    url: str
    window_start: pd.Timestamp | None = None   # filing-window start, if inferable
    window_end: pd.Timestamp | None = None     # filing-window end, if inferable


def _headers(identity: str) -> dict[str, str]:
    return {
        "User-Agent": identity,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


class SecDateParseError(ValueError):
    """Raised when an SEC date column fails deterministic parsing."""


def _sec_date(
    s: pd.Series,
    *,
    column_name: str = "date",
    max_nat_fraction: float = 0.01,
) -> pd.Series:
    """Parse SEC date columns deterministically and fail loudly on bad coverage."""
    x = s.astype("string").str.strip()
    x = x.mask(x.eq(""))
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%m-%d-%Y", "%d-%b-%Y"):
        miss = out.isna() & x.notna()
        if not miss.any():
            break
        parsed = pd.to_datetime(x.loc[miss], format=fmt, errors="coerce")
        ok = parsed.notna()
        out.loc[parsed.index[ok]] = parsed.loc[ok]

    fail = out.isna() & x.notna()
    fail_count = int(fail.sum())
    non_null = int(x.notna().sum())
    fail_fraction = (fail_count / non_null) if non_null else 0.0
    if fail_count and fail_fraction > max_nat_fraction:
        sample = x.loc[fail].drop_duplicates().head(5).tolist()
        raise SecDateParseError(
            f"{column_name}: failed to parse {fail_count}/{non_null} SEC dates "
            f"({fail_fraction:.2%}); sample={sample}"
        )
    return out


def _month_window_url(start: pd.Timestamp, end: pd.Timestamp) -> DatasetURL:
    token = f"01{start.strftime('%b').lower()}{start.year}-{end.day:02d}{end.strftime('%b').lower()}{end.year}_form13f.zip"
    return DatasetURL(SEC_13F_BASE + token, start.normalize(), end.normalize())


def _new_style_windows(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[DatasetURL]:
    """
    2024+ SEC bulk files use filing-month windows ending Feb/May/Aug/Nov.
    Examples:
      01jan2024-29feb2024_form13f.zip
      01mar2024-31may2024_form13f.zip
      01dec2024-28feb2025_form13f.zip
    """
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    out: list[DatasetURL] = []

    # First special short window: Jan-Feb 2024.
    first = _month_window_url(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-29"))
    if first.window_end >= s and first.window_start <= e:
        out.append(first)

    cur = pd.Timestamp("2024-03-01")
    while cur <= e:
        end_month = cur.month + 2
        end_year = cur.year
        if end_month > 12:
            end_month -= 12
            end_year += 1
        last_day = calendar.monthrange(end_year, end_month)[1]
        win_end = pd.Timestamp(end_year, end_month, last_day)
        if win_end >= s and cur <= e:
            out.append(_month_window_url(cur, win_end))
        cur = win_end + pd.offsets.Day(1)
    return out


def _infer_window_from_url(url: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    m = re.search(r"/(\d{2}[a-z]{3}\d{4})-(\d{2}[a-z]{3}\d{4})_form13f\.zip", url, flags=re.I)
    if m:
        try:
            return (pd.to_datetime(m.group(1), format="%d%b%Y"),
                    pd.to_datetime(m.group(2), format="%d%b%Y"))
        except Exception:
            return None, None

    m = re.search(r"/(\d{4})q([1-4])_form13f\.zip", url, flags=re.I)
    if m:
        period = pd.Period(year=int(m.group(1)), quarter=int(m.group(2)), freq="Q")
        return period.to_timestamp(how="start").normalize(), period.to_timestamp(how="end").normalize()
    return None, None


def discover_dataset_urls(identity: str, filing_start: str | pd.Timestamp, filing_end: str | pd.Timestamp) -> list[DatasetURL]:
    """
    Scrape the official SEC landing page for available 13F ZIPs. Falls back to
    generated URLs when the page is unavailable.
    """
    import requests

    fs = pd.Timestamp(filing_start).normalize()
    fe = pd.Timestamp(filing_end).normalize()
    found: dict[str, DatasetURL] = {}
    try:
        r = requests.get(SEC_13F_LANDING, headers=_headers(identity), timeout=60)
        r.raise_for_status()
        for href in re.findall(r'href=["\']([^"\']+_form13f\.zip)["\']', r.text, flags=re.I):
            url = urljoin(SEC_13F_LANDING, href)
            ws, we = _infer_window_from_url(url)
            if ws is not None and we is not None and we >= fs and ws <= fe:
                found[url] = DatasetURL(url, ws, we)
    except Exception:
        pass

    # Generated fallback for 2024+ windows. Do not generate unfinished windows;
    # if the SEC landing page does not list them yet, they are not usable inputs.
    today = pd.Timestamp.today().normalize()
    for d in _new_style_windows(max(fs, pd.Timestamp("2024-01-01")), fe):
        if d.window_end is not None and d.window_end > today:
            continue
        found.setdefault(d.url, d)

    # Legacy fallback through 2023Q4. These archives were simple YYYYqN names.
    for y, q in filing_quarters_between(fs, min(fe, pd.Timestamp("2023-12-31"))):
        ws, we = _infer_window_from_url(LEGACY_URL_PATTERN.format(y=y, q=q))
        url = LEGACY_URL_PATTERN.format(y=y, q=q)
        found.setdefault(url, DatasetURL(url, ws, we))
        old_url = OLDER_DERA_PATTERN.format(y=y, q=q)
        found.setdefault(old_url, DatasetURL(old_url, ws, we))

    return list(found.values())


def filing_quarters_between(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[tuple[int, int]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    p = pd.Period(s, "Q")
    while p.to_timestamp(how="end") >= s and p.to_timestamp(how="start") <= e:
        out.append((p.year, p.quarter))
        p += 1
    return out


def quarters_between(start: str, end: str) -> list[tuple[int, int]]:
    """Backward-compatible helper retained for older scripts."""
    return filing_quarters_between(start, end)


def report_quarter_ends_between(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[pd.Timestamp]:
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out: list[pd.Timestamp] = []
    p = pd.Period(s, "Q")
    while p.to_timestamp(how="start").normalize() <= e:
        q_end = p.to_timestamp(how="end").normalize()
        if s <= q_end <= e:
            out.append(q_end)
        p += 1
    return out


def _try_download(url: str, identity: str) -> bytes | None:
    import requests
    r = requests.get(url, headers=_headers(identity), timeout=120)
    if r.status_code == 200 and r.content[:2] == b"PK":
        return r.content
    return None


def download_quarter(year: int, q: int, identity: str) -> bytes | None:
    """
    Backward-compatible downloader for legacy callers. For 2024+ this returns the
    corresponding new-style filing-window archive where possible.
    """
    candidates: list[str] = [
        LEGACY_URL_PATTERN.format(y=year, q=q),
        OLDER_DERA_PATTERN.format(y=year, q=q),
    ]
    if year >= 2024:
        # q here is a filing-window index, not necessarily report quarter.
        # Q1 -> Jan-Feb; Q2 -> Mar-May; Q3 -> Jun-Aug; Q4 -> Sep-Nov.
        mapping = {
            1: (pd.Timestamp(year, 1, 1), pd.Timestamp(year, 2, calendar.monthrange(year, 2)[1])),
            2: (pd.Timestamp(year, 3, 1), pd.Timestamp(year, 5, 31)),
            3: (pd.Timestamp(year, 6, 1), pd.Timestamp(year, 8, 31)),
            4: (pd.Timestamp(year, 9, 1), pd.Timestamp(year, 11, 30)),
        }
        if q in mapping:
            candidates.insert(0, _month_window_url(*mapping[q]).url)
    for url in candidates:
        b = _try_download(url, identity)
        if b is not None:
            return b
        time.sleep(0.3)
    print(f"  [warn] {year}Q{q}: dataset not found at known URLs — check the SEC landing page")
    return None


def _read_tsv(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    cand = [n for n in z.namelist() if n.upper().endswith(name)]
    if not cand:
        return pd.DataFrame()
    with z.open(cand[0]) as fh:
        return pd.read_csv(fh, sep="\t", dtype=str, on_bad_lines="skip", low_memory=False)


def _col(df: pd.DataFrame, *names: str, default: str | None = None) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default] * len(df), index=df.index)


def _security_type(putcall: pd.Series, share_amount_type: pd.Series) -> pd.Series:
    pc = putcall.fillna("").astype(str).str.upper()
    amt_type = share_amount_type.fillna("").astype(str).str.upper()
    return pd.Series(
        np.select(
            [pc.str.contains("PUT"), pc.str.contains("CALL"), amt_type.eq("PRN")],
            ["PUT", "CALL", "PRN"],
            default="SH",
        ),
        index=putcall.index,
    )


def parse_quarter(zip_bytes: bytes) -> pd.DataFrame:
    """One SEC 13F ZIP -> long holdings frame."""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    sub = _read_tsv(z, "SUBMISSION.TSV")
    cov = _read_tsv(z, "COVERPAGE.TSV")
    inf = _read_tsv(z, "INFOTABLE.TSV")
    if sub.empty or inf.empty:
        return pd.DataFrame()

    sub = sub.rename(columns={c: c.upper() for c in sub.columns})
    cov = cov.rename(columns={c: c.upper() for c in cov.columns})
    inf = inf.rename(columns={c: c.upper() for c in inf.columns})

    sub = sub[sub["SUBMISSIONTYPE"].astype("string").str.contains("13F-HR", na=False)].copy()
    sub["FILING_DATE_PARSED"] = _sec_date(sub["FILING_DATE"], column_name="FILING_DATE")
    sub["PERIOD_PARSED"] = _sec_date(sub["PERIODOFREPORT"], column_name="PERIODOFREPORT")
    sub = sub.dropna(subset=["FILING_DATE_PARSED", "PERIOD_PARSED"])

    if "FILINGMANAGER_NAME" in cov.columns:
        name = cov[["ACCESSION_NUMBER", "FILINGMANAGER_NAME"]]
    else:
        name = pd.DataFrame(columns=["ACCESSION_NUMBER", "FILINGMANAGER_NAME"])

    meta = sub.merge(name, on="ACCESSION_NUMBER", how="left")
    df = inf.merge(meta, on="ACCESSION_NUMBER", how="inner")
    if df.empty:
        return pd.DataFrame()

    pc = _col(df, "PUTCALL", default="").fillna("").astype(str).str.upper()
    share_amount_type = _col(df, "SSHPRNAMTTYPE", "SSHPRNAMT_TYPE", default="SH")
    cusip = _col(df, "CUSIP").astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.zfill(9)

    out = pd.DataFrame(dict(
        cik=_col(df, "CIK").astype("string").str.zfill(10),
        manager=_col(df, "CIK").astype("string").str.zfill(10),  # stable key for engine
        manager_name=_col(df, "FILINGMANAGER_NAME").fillna(_col(df, "CIK")),
        accession_number=_col(df, "ACCESSION_NUMBER").astype("string"),
        submission_type=_col(df, "SUBMISSIONTYPE").astype("string"),
        period_date=df["PERIOD_PARSED"],
        filing_date=df["FILING_DATE_PARSED"],
        cusip=cusip,
        issuer=_col(df, "NAMEOFISSUER", default=None),
        value=pd.to_numeric(_col(df, "VALUE"), errors="coerce"),
        shares=pd.to_numeric(_col(df, "SSHPRNAMT", "SSHPRNAMT_VALUE"), errors="coerce"),
        share_amount_type=share_amount_type.fillna("").astype(str).str.upper(),
        sec_type=_security_type(pc, share_amount_type),
    )).dropna(subset=["period_date", "filing_date", "value", "cusip"])

    out = out[out["cusip"].str.len().eq(9)]

    # SEC value-unit normalization: older archives are generally in $000s; EDGAR
    # 22.4.1-era structured files use whole dollars. Verify on a sample before
    # publishing results, but this date split matches the earlier prototype.
    pre = out["filing_date"] < pd.Timestamp("2023-01-01")
    out.loc[pre, "value"] = out.loc[pre, "value"] * 1000.0
    return out


# --------------------------------------------------------------------------- #
def coarse_prefilter(holdings: pd.DataFrame,
                     min_aum=1e9, max_aum=30e9, max_holdings=60,
                     max_put_weight=0.10) -> pd.DataFrame:
    """Coarse, point-in-time filing-level screen before the engine's final screen."""
    h = holdings.copy()
    g = h[h.sec_type == "SH"].groupby(["cik", "period_date"])
    stats = g.agg(aum=("value", "sum"), n=("cusip", "nunique")).reset_index()
    tot = h.groupby(["cik", "period_date"])["value"].sum().rename("tot").reset_index()
    putv = (h[h.sec_type == "PUT"].groupby(["cik", "period_date"])["value"].sum()
            .rename("putv").reset_index())
    stats = (stats.merge(tot, on=["cik", "period_date"], how="left")
                  .merge(putv, on=["cik", "period_date"], how="left")
                  .fillna({"putv": 0}))
    stats["put_w"] = stats["putv"] / stats["tot"].replace(0, np.nan)
    ok = stats[(stats.aum.between(min_aum, max_aum)) &
               (stats.n <= max_holdings) &
               (stats.put_w.fillna(0) <= max_put_weight)][["cik", "period_date"]]
    kept = h.merge(ok, on=["cik", "period_date"], how="inner")
    n_before = holdings[["cik", "period_date"]].drop_duplicates().shape[0]
    n_after = ok.shape[0]
    print(f"  coarse prefilter: {n_before} -> {n_after} filings "
          f"({kept['cik'].nunique()} distinct filers survive)")
    return kept


def build_holdings_universe(start: str, end: str, identity: str,
                            cache_dir: str | None = "13f_cache", **prefilter_kw) -> pd.DataFrame:
    """
    Download all SEC bulk archives needed for report periods in [start, end].
    Returns standardized holdings ready for engine.run_backtest().
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    report_start = pd.Timestamp(start).normalize()
    report_end = pd.Timestamp(end).normalize()
    report_periods = report_quarter_ends_between(report_start, report_end)
    if not report_periods:
        raise ValueError(f"No quarter-end 13F report periods in requested range: {start} to {end}")

    # 13F filings arrive after quarter-end. Use the actual target quarter-ends,
    # not the raw start date, or arbitrary Jan/Apr/Jul starts pull unrelated
    # filing-window archives.
    filing_start = min(report_periods)
    filing_end = max(report_periods) + pd.Timedelta(days=75)
    print(
        "  target report periods: "
        + ", ".join(p.strftime("%Y-%m-%d") for p in report_periods)
    )
    print(f"  filing availability window: {filing_start.date()} to {filing_end.date()}")

    datasets = discover_dataset_urls(identity, filing_start, filing_end)
    if not datasets:
        raise RuntimeError("No SEC 13F dataset URLs discovered.")
    print(f"  SEC datasets selected: {len(datasets)}")

    frames = []
    seen_urls: set[str] = set()
    for d in datasets:
        if d.url in seen_urls:
            continue
        seen_urls.add(d.url)
        fname = d.url.rsplit("/", 1)[-1].replace("_form13f.zip", "")
        cpath = os.path.join(cache_dir, f"{fname}.{PARSED_CACHE_VERSION}.parquet") if cache_dir else None
        if cpath and os.path.exists(cpath):
            dfq = pd.read_parquet(cpath)
            print(f"[{fname}] cache hit: {len(dfq)} rows")
            frames.append(dfq)
            continue
        print(f"[{fname}] downloading …")
        b = _try_download(d.url, identity)
        if b is None:
            print(f"  [warn] dataset not found or blocked: {d.url}")
            continue
        dfq = parse_quarter(b)
        if not dfq.empty:
            dfq = dfq[dfq["period_date"].between(report_start, report_end)]
        print(f"[{fname}] parsed: {len(dfq)} in-range rows")
        if cpath and not dfq.empty:
            dfq.to_parquet(cpath)
            print(f"[{fname}] cached: {cpath}")
        frames.append(dfq)
        time.sleep(0.5)

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError("No 13F datasets downloaded/parsed — verify SEC URLs and User-Agent identity.")

    allh = pd.concat(frames, ignore_index=True)
    print(f"  raw in-range holdings rows before de-dup: {len(allh)}")
    # Final guard: overlapping archives can repeat the same filing; keep distinct
    # filing versions so amendments remain point-in-time events for the engine.
    allh = (allh.sort_values(["cik", "period_date", "filing_date"])
                .drop_duplicates(["accession_number", "cusip", "sec_type"], keep="last"))
    print(f"  holdings rows after accession/CUSIP/sec_type de-dup: {len(allh)}")
    return coarse_prefilter(allh, **prefilter_kw)
