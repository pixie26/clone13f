"""
Data adapters — the ONLY part that touches the network. Run on your infra
(open internet); everything in engine.py is pure pandas and needs none of this.

Each function returns the standardized frames engine.py expects. All sources here
are public + free. Frictions are flagged inline with  ## FRICTION.

Deps:  pip install edgartools yfinance requests
NOTE:  edgartools' attribute surface drifts across versions — the 13F parser
       below is written defensively; verify column names against YOUR installed
       version once (one print of f.infotable.columns) and adjust the mapping.
"""
from __future__ import annotations
import contextlib
import io
from pathlib import Path
import re
import time
import zipfile
import numpy as np
import pandas as pd
import requests

PRICE_ELIGIBLE_SEC_TYPES = {"SH"}
YFINANCE_TICKER_RE = r"^[A-Z]{1,6}([.\-][A-Z]{1,2})?$"
OPENFIGI_US_EQUITY_FILTERS = {
    "currency": "USD",
    "marketSecDes": "Equity",
    "exchCode": "US",
}


def _is_yfinance_ticker(value) -> bool:
    if pd.isna(value):
        return False
    return bool(re.fullmatch(YFINANCE_TICKER_RE, str(value).strip().upper()))


def _is_us_openfigi_exchange(value) -> bool:
    exch = str(value or "").strip().upper()
    return exch == "US" or (len(exch) == 2 and exch.startswith("U"))


def _select_openfigi_ticker(data: list[dict] | None) -> str | None:
    if not data:
        return None
    candidates = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        ticker = rec.get("ticker")
        if not _is_yfinance_ticker(ticker):
            continue
        market_sector = str(rec.get("marketSector", "")).upper()
        exch = str(rec.get("exchCode", "")).upper()
        security_type = " ".join(
            str(rec.get(k, ""))
            for k in ("securityType", "securityType2", "securityDescription")
        ).upper()
        if market_sector and market_sector != "EQUITY":
            continue
        if exch and not _is_us_openfigi_exchange(exch):
            continue
        if any(bad in security_type for bad in ("OPTION", "FUTURE", "CORP", "BOND", "NOTE", "DEBT")):
            continue
        score = 0
        if _is_us_openfigi_exchange(exch):
            score += 2
        if market_sector == "EQUITY":
            score += 1
        candidates.append((score, str(ticker).strip().upper()))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _load_openfigi_cache(cache_path: str | Path | None) -> dict[str, str | None]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "cusip" not in df or "ticker" not in df:
        return {}
    out: dict[str, str | None] = {}
    for row in df[["cusip", "ticker"]].itertuples(index=False):
        cusip = str(row.cusip).strip().upper()
        ticker = None if pd.isna(row.ticker) else str(row.ticker).strip().upper()
        if ticker is not None and not _is_yfinance_ticker(ticker):
            ticker = None
        out[cusip] = ticker
    return out


def _write_openfigi_cache(cache_path: str | Path | None, cache: dict[str, str | None]) -> None:
    if cache_path is None:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"cusip": c, "ticker": t} for c, t in sorted(cache.items())]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _normalise_close_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"]
    elif "Close" in raw.columns:
        close = raw["Close"]
    else:
        return pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.columns = [str(c).strip().upper() for c in close.columns]
    return close


def _yf_download_close(yf, batch: list[str], start: str, end: str) -> pd.DataFrame:
    kwargs = dict(start=start, end=end, auto_adjust=True, progress=False, threads=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw = yf.download(batch, timeout=30, **kwargs)
    except TypeError:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw = yf.download(batch, **kwargs)
    return _normalise_close_frame(raw, batch)


def _looks_rate_limited(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "ratelimit" in text or "rate limit" in text or "too many requests" in text or "429" in text


# --------------------------------------------------------------------------- #
# 1) 13F holdings from EDGAR  (public, free; retains dead filers => no survivorship)
# --------------------------------------------------------------------------- #
def fetch_13f(filers: dict[str, str], start: str, end: str, identity: str) -> pd.DataFrame:
    """
    filers   : {manager_name: cik}    cik as string/int, e.g. {"AKRE":"0001112520"}
    identity : "Name you@firm.com"     SEC requires a UA on every request.
    Returns long holdings frame: manager, period_date, filing_date, ticker(=CUSIP here),
            value, shares, issuer, sec_type.  (CUSIP->ticker done separately.)
    """
    from edgar import set_identity, Company
    set_identity(identity)
    out = []
    for name, cik in filers.items():
        for filing in Company(str(cik)).get_filings(form="13F-HR"):
            fdate = pd.Timestamp(filing.filing_date)
            if not (pd.Timestamp(start) <= fdate <= pd.Timestamp(end)):
                continue
            obj = filing.obj()
            tbl = getattr(obj, "infotable", None)
            if tbl is None:
                tbl = getattr(obj, "information_table", None)
            if tbl is None or len(tbl) == 0:
                continue
            tbl = pd.DataFrame(tbl)
            cols = {c.lower(): c for c in tbl.columns}
            def col(*names, default=None):
                for n in names:
                    if n in cols: return tbl[cols[n]]
                return pd.Series([default] * len(tbl))
            period = pd.Timestamp(getattr(obj, "report_period", None) or
                                  getattr(filing, "report_date", None) or fdate)
            putcall = col("putcall", "put/call").fillna("").astype(str).str.upper()
            sec_type = np.where(putcall.str.contains("PUT"), "PUT",
                       np.where(putcall.str.contains("CALL"), "CALL", "SH"))
            out.append(pd.DataFrame(dict(
                manager=name, period_date=period.normalize(),
                filing_date=fdate.normalize(),
                cusip=col("cusip").astype(str).str.zfill(9),
                issuer=col("nameofissuer", "issuer"),
                value=pd.to_numeric(col("value"), errors="coerce"),
                shares=pd.to_numeric(col("sharesprnamount", "shares"), errors="coerce"),
                sec_type=sec_type,
            )))
            time.sleep(0.15)   ## FRICTION: SEC fair-access ~10 req/s; be polite
    df = pd.concat(out, ignore_index=True)
    ## FRICTION: pre-2023 'value' was reported in $thousands; post-2022Q3 in $.
    ##           Normalize before use (check filing date vs the rule change).
    return df


# --------------------------------------------------------------------------- #
# 2) CUSIP -> ticker via OpenFIGI  (free; 25 req/min anon, higher with free key)
# --------------------------------------------------------------------------- #
def cusip_to_ticker(
    cusips,
    api_key: str | None = None,
    *,
    cache_path: str | Path | None = None,
) -> dict[str, str]:
    ## FRICTION: this is THE assembly bottleneck. Cache the map to disk and only
    ##           query new CUSIPs. ~25 req/min without a (free) OpenFIGI key.
    url = "https://api.openfigi.com/v3/mapping"
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["X-OPENFIGI-APIKEY"] = api_key
    uniq = sorted({str(c).strip().upper() for c in cusips if pd.notna(c)})
    cache = _load_openfigi_cache(cache_path)
    cached = {c: cache[c] for c in uniq if c in cache}
    mp: dict[str, str] = {c: t for c, t in cached.items() if t is not None}
    todo = [c for c in uniq if c not in cache]
    if cache_path is not None:
        print(f"  OpenFIGI cache: {len(cached)}/{len(uniq)} CUSIPs found at {cache_path}")
    n_batches = int(np.ceil(len(todo) / 100)) if todo else 0
    print(f"  OpenFIGI mapping: {len(todo)} uncached CUSIPs in {n_batches} batches")
    for i in range(0, len(todo), 100):
        batch_cusips = todo[i:i + 100]
        batch_no = i // 100 + 1
        print(f"    OpenFIGI batch {batch_no}/{n_batches}: {batch_cusips[0]}..{batch_cusips[-1]}")
        batch = [
            {"idType": "ID_CUSIP", "idValue": c, **OPENFIGI_US_EQUITY_FILTERS}
            for c in batch_cusips
        ]
        r = requests.post(url, json=batch, headers=hdr, timeout=30)
        if r.status_code == 429:
            print("    [warn] OpenFIGI rate limited; sleeping 6s then retrying batch")
            time.sleep(6); r = requests.post(url, json=batch, headers=hdr, timeout=30)
        r.raise_for_status()
        batch_mapped = 0
        batch_rejected = 0
        for c, res in zip(batch_cusips, r.json()):
            data = res.get("data") if isinstance(res, dict) else None
            ticker = _select_openfigi_ticker(data)
            if ticker is not None:
                mp[c] = ticker
                cache[c] = ticker
                batch_mapped += 1
            elif data:
                cache[c] = None
                batch_rejected += 1
            else:
                cache[c] = None
        print(
            f"    OpenFIGI batch {batch_no}/{n_batches}: "
            f"mapped {batch_mapped}/{len(batch_cusips)}, rejected_non_yahoo_us {batch_rejected}"
        )
        _write_openfigi_cache(cache_path, cache)
        time.sleep(2.5 if not api_key else 0.3)
    _write_openfigi_cache(cache_path, cache)
    print(f"  OpenFIGI mapping complete: {len(mp)}/{len(uniq)} CUSIPs mapped")
    return mp


def mapping_diagnostics(holdings: pd.DataFrame, cmap: dict[str, str]) -> dict:
    mapped = holdings["cusip"].map(cmap).notna()
    total_value = holdings["value"].sum()
    mapped_value = holdings.loc[mapped, "value"].sum()
    return {
        "rows_total": int(len(holdings)),
        "rows_mapped": int(mapped.sum()),
        "rows_unmapped": int((~mapped).sum()),
        "cusips_total": int(holdings["cusip"].nunique()),
        "cusips_mapped": int(holdings.loc[mapped, "cusip"].nunique()),
        "cusips_unmapped": int(holdings.loc[~mapped, "cusip"].nunique()),
        "value_total": float(total_value),
        "value_mapped": float(mapped_value),
        "value_coverage": float(mapped_value / total_value) if total_value else float("nan"),
        "drop_reason": "unmapped_cusip",
    }


def map_holdings_to_tickers(
    holdings: pd.DataFrame,
    cmap: dict[str, str],
    *,
    min_value_coverage: float = 0.90,
    strict: bool = False,
) -> pd.DataFrame:
    h = holdings.copy()
    h["ticker"] = h["cusip"].map(cmap).astype("string").str.strip()
    h["ticker"] = h["ticker"].mask(h["ticker"].eq(""))
    diag = mapping_diagnostics(h, cmap)
    msg = (
        "CUSIP mapping coverage: "
        f"{diag['cusips_mapped']}/{diag['cusips_total']} CUSIPs, "
        f"{diag['rows_mapped']}/{diag['rows_total']} rows, "
        f"{diag['value_coverage']:.1%} value mapped"
    )
    if strict and diag["value_coverage"] < min_value_coverage:
        raise ValueError(msg)
    if diag["rows_unmapped"]:
        print(f"  [warn] {msg}; dropping unmapped_cusip rows")
    else:
        print(f"  {msg}")
    out = h.dropna(subset=["ticker"]).copy()
    out.attrs["mapping_diagnostics"] = diag
    return out


def priceable_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Keep holdings that can plausibly be priced as exchange-traded equities by
    yfinance. 13F PRN rows are usually bonds/convertibles and OpenFIGI may map
    them to descriptions like 'ABC 2.25 08/15/28', not stock symbols.
    """
    h = holdings.copy()
    sec_type = h.get("sec_type", pd.Series("SH", index=h.index)).fillna("SH").astype(str).str.upper()
    ticker = h["ticker"].astype("string").str.strip().str.upper()
    keep_sec = sec_type.isin(PRICE_ELIGIBLE_SEC_TYPES)
    keep_ticker = ticker.map(_is_yfinance_ticker).astype(bool)
    out = h.loc[keep_sec & keep_ticker].copy()
    out["ticker"] = ticker.loc[out.index]
    dropped = int(len(h) - len(out))
    if dropped:
        print(
            "  [warn] price input filter: "
            f"dropped {dropped}/{len(h)} mapped rows with non-equity sec_type or non-yfinance ticker"
        )
    out.attrs.update(h.attrs)
    out.attrs["price_filter_diagnostics"] = {
        "rows_total": int(len(h)),
        "rows_priceable": int(len(out)),
        "rows_dropped": dropped,
        "drop_reason": "non_equity_or_invalid_yfinance_ticker",
    }
    return out


def align_holdings_to_prices(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    min_value_coverage: float = 0.80,
    strict: bool = False,
) -> pd.DataFrame:
    price_tickers = pd.Index(prices.columns.astype(str).str.upper())
    h = holdings.copy()
    h["ticker"] = h["ticker"].astype("string").str.strip().str.upper()
    keep = h["ticker"].isin(price_tickers)
    total_value = h["value"].sum()
    kept_value = h.loc[keep, "value"].sum()
    diag = {
        "rows_total": int(len(h)),
        "rows_with_prices": int(keep.sum()),
        "rows_without_prices": int((~keep).sum()),
        "tickers_total": int(h["ticker"].nunique()),
        "tickers_with_prices": int(h.loc[keep, "ticker"].nunique()),
        "tickers_without_prices": int(h.loc[~keep, "ticker"].nunique()),
        "value_total": float(total_value),
        "value_with_prices": float(kept_value),
        "value_coverage": float(kept_value / total_value) if total_value else float("nan"),
        "drop_reason": "no_yfinance_price_coverage",
    }
    msg = (
        "price alignment coverage: "
        f"{diag['tickers_with_prices']}/{diag['tickers_total']} tickers, "
        f"{diag['rows_with_prices']}/{diag['rows_total']} rows, "
        f"{diag['value_coverage']:.1%} value retained"
    )
    if strict and diag["value_coverage"] < min_value_coverage:
        raise ValueError(msg)
    if diag["rows_without_prices"]:
        print(f"  [warn] {msg}; dropping no_yfinance_price_coverage rows")
    else:
        print(f"  {msg}")
    out = h.loc[keep].copy()
    out.attrs.update(h.attrs)
    out.attrs["price_alignment_diagnostics"] = diag
    return out


# --------------------------------------------------------------------------- #
# 3) Monthly stock returns  (yfinance free; prefer CRSP if you have WRDS)
# --------------------------------------------------------------------------- #
def fetch_prices(
    tickers,
    start: str,
    end: str,
    *,
    batch_size: int = 50,
    max_retries: int = 2,
    rate_limit_sleep: int = 60,
) -> pd.DataFrame:
    ## FRICTION: yfinance has survivorship gaps (delisted names vanish). For a
    ##           publishable backtest use CRSP via WRDS; yfinance is fine first-pass.
    import yfinance as yf
    raw_unique = sorted({str(t).strip().upper() for t in tickers if pd.notna(t)})
    clean = sorted({t for t in raw_unique if _is_yfinance_ticker(t)})
    if not clean:
        raise ValueError("No valid yfinance tickers to download.")
    dropped = len(raw_unique) - len(clean)
    if dropped:
        print(f"  [warn] price download: skipped {dropped} invalid ticker strings")

    frames: list[pd.DataFrame] = []
    failed_batches: list[str] = []
    n_batches = int(np.ceil(len(clean) / batch_size))
    print(f"  price download: {len(clean)} tickers in {n_batches} batches")
    for i in range(0, len(clean), batch_size):
        batch = clean[i:i + batch_size]
        batch_no = i // batch_size + 1
        print(f"    yfinance batch {batch_no}/{n_batches}: {batch[0]}..{batch[-1]}")
        for attempt in range(max_retries + 1):
            try:
                close = _yf_download_close(yf, batch, start, end)
                if not close.empty:
                    frames.append(close)
                break
            except Exception as exc:
                last_attempt = attempt >= max_retries
                if _looks_rate_limited(exc) and not last_attempt:
                    wait = rate_limit_sleep * (attempt + 1)
                    print(
                        "    [warn] yfinance rate limited; "
                        f"sleeping {wait}s before retry {attempt + 1}/{max_retries}"
                    )
                    time.sleep(wait)
                    continue
                failed_batches.append(f"{batch[0]}..{batch[-1]}: {exc}")
                print(f"    [warn] yfinance batch failed: {batch[0]}..{batch[-1]}: {exc}")
                break
        time.sleep(0.1)

    if not frames:
        detail = f"; failed batches: {failed_batches[:3]}" if failed_batches else ""
        raise RuntimeError(f"No price data downloaded from yfinance{detail}")

    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()].sort_index()
    missing_cols = sorted(set(clean) - set(px.columns))
    all_nan_cols = sorted(px.columns[px.isna().all()].tolist())
    no_price = sorted(set(missing_cols).union(all_nan_cols))
    if no_price:
        px = px.drop(columns=[c for c in no_price if c in px.columns])
        sample = ", ".join(no_price[:20])
        print(f"  [warn] price coverage: no yfinance close data for {len(no_price)}/{len(clean)} tickers; sample: {sample}")
    if px.empty:
        raise RuntimeError("All downloaded yfinance price columns were empty.")

    monthly = px.resample("ME").last()
    returns = monthly.pct_change(fill_method=None).iloc[1:]
    returns.attrs["price_diagnostics"] = {
        "tickers_requested": int(len(clean)),
        "tickers_with_close": int(px.shape[1]),
        "tickers_without_close": int(len(no_price)),
        "invalid_tickers_skipped": int(dropped),
        "failed_batches": failed_batches[:10],
    }
    return returns


def _parse_ken_french_monthly_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(",") and any(token in stripped for token in ("Mkt-RF", "Mom")):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not locate monthly Ken French table header.")

    data_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        first = line.split(",", 1)[0].strip()
        if not first or not re.fullmatch(r"\d{6}", first):
            break
        data_lines.append(line)
    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df = df.rename(columns={df.columns[0]: "date"})
    df.columns = [str(c).strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date")
    return df.apply(pd.to_numeric, errors="coerce") / 100.0


def _fetch_ken_french_monthly_zip(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV found in Ken French ZIP: {url}")
        text = z.read(names[0]).decode("latin1")
    return _parse_ken_french_monthly_csv(text)


# --------------------------------------------------------------------------- #
# 4) Fama-French 5 + Momentum  (Ken French library, public + free)
# --------------------------------------------------------------------------- #
def fetch_factors(start: str, end: str) -> pd.DataFrame:
    ff5 = _fetch_ken_french_monthly_zip(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )
    mom = _fetch_ken_french_monthly_zip(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_CSV.zip"
    )
    mom_col = next((c for c in mom.columns if c.lower().startswith("mom")), mom.columns[0])
    f = ff5.rename(columns={"Mkt-RF": "MKT"}).join(mom[[mom_col]].rename(columns={mom_col: "MOM"}))
    f = f[["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]]
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return f.loc[(f.index >= s) & (f.index <= e)]


# --------------------------------------------------------------------------- #
# 5) Manager classification (filer type + long-bias)
# --------------------------------------------------------------------------- #
# ## FRICTION: deciding "is this filer a long-biased fundamental HF?" cleanly needs
# ##   HFR/TASS/CISDM (paid). The free path: SEC IAPD Form ADV (public) gives the
# ##   firm's reported strategy + AUM; combine with two 13F-derived heuristics that
# ##   engine.py already applies (low listed-put weight => not heavily hedged; stable
# ##   concentrated book => fundamental, not quant). For a first pass, hand-curate the
# ##   `filers` dict to long-biased shops and let engine.py's characteristic filters
# ##   do the rest — but remember a hand-picked dict reintroduces selection bias, so
# ##   for the headline run, build `filers` from a RULE (e.g. all 13F filers in the
# ##   AUM band whose ADV strategy == long/long-short equity) applied point-in-time.
