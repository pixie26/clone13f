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
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns:
        close = raw["Close"]
    else:
        return pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close.columns = [str(c).strip().upper() for c in close.columns]
    return close


def _yf_download_close(yf, batch: list[str], start: str, end: str) -> pd.DataFrame:
    kwargs = dict(start=start, end=end, auto_adjust=True, progress=False, threads=True, group_by="column")
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


def _yfinance_probe(yf, start: str, end: str) -> str:
    try:
        close = _yf_download_close(yf, ["SPY", "IWD", "AAPL"], start, end)
    except Exception as exc:
        return f"probe_exception={type(exc).__name__}: {exc}"
    if close.empty:
        return "probe_empty_close"
    non_empty = close.dropna(axis=1, how="all").columns.tolist()
    if not non_empty:
        return "probe_all_nan_close"
    return f"probe_ok={','.join(map(str, non_empty))}"


def _yahoo_chart_symbol(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def _yahoo_chart_download_one(ticker: str, start: str, end: str) -> pd.Series:
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    # Yahoo chart period2 is exclusive. Add one day so a requested month-end is
    # available for monthly return construction when the market was open.
    end_ts = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    symbol = _yahoo_chart_symbol(ticker)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers, timeout=20)
    if response.status_code in {404, 410}:
        return pd.Series(dtype="float64", name=ticker)
    if response.status_code == 429:
        raise RuntimeError(f"Yahoo Chart API rate limited for {ticker}")
    response.raise_for_status()
    payload = response.json().get("chart", {})
    if payload.get("error"):
        return pd.Series(dtype="float64", name=ticker)
    results = payload.get("result") or []
    if not results:
        return pd.Series(dtype="float64", name=ticker)
    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    adjclose = ((indicators.get("adjclose") or [{}])[0] or {}).get("adjclose")
    close = ((indicators.get("quote") or [{}])[0] or {}).get("close")
    values = adjclose or close
    if not timestamps or not values:
        return pd.Series(dtype="float64", name=ticker)
    idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
    return pd.Series(values, index=idx, name=ticker, dtype="float64").dropna()


def _yahoo_chart_download_close(
    batch: list[str],
    start: str,
    end: str,
    *,
    max_workers: int = 8,
) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not batch:
        return pd.DataFrame()
    workers = max(1, min(max_workers, len(batch)))
    series: list[pd.Series] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ticker = {
            executor.submit(_yahoo_chart_download_one, ticker, start, end): ticker
            for ticker in batch
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                item = future.result()
            except Exception as exc:
                errors.append(f"{ticker}: {type(exc).__name__}: {exc}")
                continue
            if not item.empty:
                series.append(item)
    if errors:
        print(f"    [warn] Yahoo Chart API ticker errors: {errors[:5]}")
    if not series:
        return pd.DataFrame()
    close = pd.concat(series, axis=1)
    close.columns = [str(c).strip().upper() for c in close.columns]
    return close.sort_index()


def _yahoo_chart_probe(start: str, end: str) -> str:
    try:
        close = _yahoo_chart_download_close(["SPY", "IWD", "AAPL"], start, end, max_workers=3)
    except Exception as exc:
        return f"chart_probe_exception={type(exc).__name__}: {exc}"
    if close.empty:
        return "chart_probe_empty_close"
    non_empty = close.dropna(axis=1, how="all").columns.tolist()
    if not non_empty:
        return "chart_probe_all_nan_close"
    return f"chart_probe_ok={','.join(map(str, non_empty))}"


def _fetch_yahoo_chart_batches(
    tickers: list[str],
    start: str,
    end: str,
    *,
    batch_size: int,
    cache_path: str | Path | None,
    max_workers: int,
) -> tuple[list[pd.DataFrame], list[str]]:
    frames: list[pd.DataFrame] = []
    failed_batches: list[str] = []
    n_batches = int(np.ceil(len(tickers) / batch_size)) if tickers else 0
    print(f"  Yahoo Chart fallback: {len(tickers)} tickers in {n_batches} batches")
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_no = i // batch_size + 1
        print(f"    yahoo-chart batch {batch_no}/{n_batches}: {batch[0]}..{batch[-1]}")
        close = _yahoo_chart_download_close(batch, start, end, max_workers=max_workers)
        non_nan_close = close.dropna(axis=1, how="all") if not close.empty else close
        if non_nan_close.empty:
            failed_batches.append(f"{batch[0]}..{batch[-1]}: empty_close")
        else:
            frames.append(non_nan_close)
            _write_price_cache(cache_path, non_nan_close)
            _write_price_coverage(cache_path, non_nan_close.columns, start, end, "fetched")
            no_close_in_batch = sorted(set(batch) - set(non_nan_close.columns))
            _write_price_coverage(cache_path, no_close_in_batch, start, end, "no_close")
        time.sleep(0.1)
    return frames, failed_batches


def _load_price_cache(cache_path: str | Path | None, start: str, end: str) -> pd.DataFrame:
    if cache_path is None:
        return pd.DataFrame()
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        px = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if px.empty:
        return pd.DataFrame()
    px.index = pd.to_datetime(px.index)
    px.columns = [str(c).strip().upper() for c in px.columns]
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return px.loc[(px.index >= s) & (px.index <= e)]


def _price_coverage_cache_path(cache_path: str | Path | None) -> Path | None:
    if cache_path is None:
        return None
    path = Path(cache_path)
    return path.with_name(path.stem + "_coverage" + path.suffix)


def _load_price_coverage(cache_path: str | Path | None) -> pd.DataFrame:
    path = _price_coverage_cache_path(cache_path)
    if path is None or not path.exists():
        return pd.DataFrame(columns=["ticker", "start", "end", "status"])
    try:
        cov = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=["ticker", "start", "end", "status"])
    if cov.empty:
        return pd.DataFrame(columns=["ticker", "start", "end", "status"])
    cov = cov.copy()
    cov["ticker"] = cov["ticker"].astype(str).str.strip().str.upper()
    cov["start"] = pd.to_datetime(cov["start"])
    cov["end"] = pd.to_datetime(cov["end"])
    cov["status"] = cov["status"].astype(str)
    return cov


def _write_price_coverage(cache_path: str | Path | None, tickers, start: str, end: str, status: str) -> None:
    path = _price_coverage_cache_path(cache_path)
    if path is None:
        return
    clean = sorted({str(t).strip().upper() for t in tickers if pd.notna(t)})
    if not clean:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_price_coverage(cache_path)
    new = pd.DataFrame({
        "ticker": clean,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "status": status,
    })
    out = new if current.empty else pd.concat([current, new], ignore_index=True)
    out = (out.sort_values(["ticker", "start", "end", "status"])
              .drop_duplicates(["ticker", "start", "end", "status"], keep="last"))
    out.to_parquet(path, index=False)


def _coverage_covers(cov: pd.DataFrame, tickers, start: str, end: str, statuses=("fetched", "no_close")) -> set[str]:
    if cov.empty:
        return set()
    clean = sorted({str(t).strip().upper() for t in tickers if pd.notna(t)})
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    hit = cov[
        cov["ticker"].isin(clean)
        & cov["status"].isin(statuses)
        & (cov["start"] <= s)
        & (cov["end"] >= e)
    ]
    return set(hit["ticker"].tolist())


def _series_spans_requested_window(s: pd.Series, start: str, end: str) -> bool:
    x = s.dropna()
    if x.empty:
        return False
    monthly = x.resample("ME").last()
    if monthly.empty:
        return False
    first_required = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")
    if len(first_required) == 0:
        return False
    return monthly.index.min() <= first_required[0] and monthly.index.max() >= first_required[-1]


def _write_price_cache(cache_path: str | Path | None, px: pd.DataFrame) -> None:
    if cache_path is None or px.empty:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame()
    if path.exists():
        try:
            current = pd.read_parquet(path)
        except Exception:
            current = pd.DataFrame()
    if not current.empty:
        current.index = pd.to_datetime(current.index)
        current.columns = [str(c).strip().upper() for c in current.columns]
    px = px.copy()
    px.index = pd.to_datetime(px.index)
    px.columns = [str(c).strip().upper() for c in px.columns]
    merged = pd.concat([current, px], axis=1) if not current.empty else px
    merged = merged.loc[:, ~merged.columns.duplicated()].sort_index()
    merged.to_parquet(path)


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
    cache_path: str | Path | None = None,
    max_consecutive_empty_batches: int = 3,
    use_chart_fallback: bool = True,
    chart_fallback_workers: int = 8,
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

    cache_px = _load_price_cache(cache_path, start, end)
    coverage = _load_price_coverage(cache_path)
    coverage_fetched = _coverage_covers(coverage, clean, start, end, statuses=("fetched",))
    coverage_no_close = _coverage_covers(coverage, clean, start, end, statuses=("no_close",))
    cache_symbol_cols = sorted(set(clean).intersection(cache_px.columns)) if not cache_px.empty else []
    cached_cols = sorted(
        t for t in cache_symbol_cols
        if t in coverage_fetched or _series_spans_requested_window(cache_px[t], start, end)
    )
    stale_cached_cols = sorted(set(cache_symbol_cols) - set(cached_cols))
    frames: list[pd.DataFrame] = []
    if cached_cols:
        cached_px = cache_px[cached_cols].dropna(axis=1, how="all")
        if not cached_px.empty:
            frames.append(cached_px)
        print(f"  yfinance cache: {len(cached_cols)}/{len(clean)} coverage-valid tickers found at {cache_path}")
    if stale_cached_cols:
        sample = ", ".join(stale_cached_cols[:20])
        print(
            "  [info] yfinance cache: "
            f"refetching {len(stale_cached_cols)} tickers whose cached history does not cover "
            f"{start} to {end}; sample: {sample}"
        )
    if coverage_no_close:
        sample = ", ".join(sorted(coverage_no_close)[:20])
        print(f"  yfinance cache: {len(coverage_no_close)} tickers previously fetched with no close data; sample: {sample}")

    todo = sorted(set(clean) - set(cached_cols) - set(coverage_no_close))
    failed_batches: list[str] = []
    empty_batches: list[str] = []
    empty_batch_tickers: list[list[str]] = []
    chart_failed_batches: list[str] = []
    used_chart_fallback = False
    consecutive_empty = 0
    n_batches = int(np.ceil(len(todo) / batch_size)) if todo else 0
    print(f"  price download: {len(todo)} uncached / {len(clean)} requested tickers in {n_batches} batches")
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        batch_no = i // batch_size + 1
        print(f"    yfinance batch {batch_no}/{n_batches}: {batch[0]}..{batch[-1]}")
        for attempt in range(max_retries + 1):
            try:
                close = _yf_download_close(yf, batch, start, end)
                if not close.empty:
                    non_nan_close = close.dropna(axis=1, how="all")
                    if non_nan_close.empty:
                        empty_batches.append(f"{batch[0]}..{batch[-1]}: all_nan_close")
                        empty_batch_tickers.append(batch)
                        consecutive_empty += 1
                    else:
                        frames.append(non_nan_close)
                        _write_price_cache(cache_path, non_nan_close)
                        _write_price_coverage(cache_path, non_nan_close.columns, start, end, "fetched")
                        no_close_in_batch = sorted(set(batch) - set(non_nan_close.columns))
                        _write_price_coverage(cache_path, no_close_in_batch, start, end, "no_close")
                        consecutive_empty = 0
                    break
                last_attempt = attempt >= max_retries
                if not last_attempt:
                    wait = min(15 * (attempt + 1), rate_limit_sleep)
                    print(
                        "    [warn] yfinance returned empty Close frame; "
                        f"sleeping {wait}s before retry {attempt + 1}/{max_retries}"
                    )
                    time.sleep(wait)
                    continue
                empty_batches.append(f"{batch[0]}..{batch[-1]}: empty_close")
                empty_batch_tickers.append(batch)
                consecutive_empty += 1
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
        if consecutive_empty >= max_consecutive_empty_batches:
            probe = _yfinance_probe(yf, start, end)
            if "probe_ok=" not in probe:
                if not use_chart_fallback:
                    raise RuntimeError(
                        "Stopping yfinance download after "
                        f"{consecutive_empty} consecutive empty batches; {probe}; "
                        f"recent empty batches: {empty_batches[-max_consecutive_empty_batches:]}"
                    )
                chart_probe = _yahoo_chart_probe(start, end)
                if "chart_probe_ok=" not in chart_probe:
                    raise RuntimeError(
                        "Stopping yfinance download after "
                        f"{consecutive_empty} consecutive empty batches; {probe}; "
                        f"Yahoo Chart fallback unavailable: {chart_probe}; "
                        f"recent empty batches: {empty_batches[-max_consecutive_empty_batches:]}"
                    )
                prior_empty = [t for batch_tickers in empty_batch_tickers for t in batch_tickers]
                remaining = todo[i + batch_size:]
                fallback_tickers = sorted(set(prior_empty).union(remaining))
                print(
                    "    [warn] yfinance returned consecutive empty batches and probe failed; "
                    f"{probe}. Switching to Yahoo Chart API fallback; {chart_probe}"
                )
                chart_frames, chart_failed_batches = _fetch_yahoo_chart_batches(
                    fallback_tickers,
                    start,
                    end,
                    batch_size=batch_size,
                    cache_path=cache_path,
                    max_workers=chart_fallback_workers,
                )
                frames.extend(chart_frames)
                used_chart_fallback = True
                break
            print(
                "    [warn] consecutive empty yfinance batches but probe succeeded; "
                "continuing and treating those tickers as uncovered"
            )
            consecutive_empty = 0
        time.sleep(0.1)

    if not frames:
        probe = _yfinance_probe(yf, start, end)
        detail_parts = []
        if failed_batches:
            detail_parts.append(f"failed batches: {failed_batches[:3]}")
        if empty_batches:
            detail_parts.append(f"empty batches: {empty_batches[:3]}")
        detail_parts.append(probe)
        raise RuntimeError("No price data downloaded from yfinance or fallback; " + "; ".join(detail_parts))

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
    # Keep partial-history tickers. Dropping any column with a single NaN creates
    # a complete-window survivor universe. The engine handles per-month gaps with
    # missing_price_policy, and rebalance logic filters names that are not priced
    # on the rebalance month.
    all_nan_return_cols = sorted(returns.columns[returns.isna().all()].tolist())
    if all_nan_return_cols:
        returns = returns.drop(columns=all_nan_return_cols)
        sample = ", ".join(all_nan_return_cols[:20])
        print(
            "  [warn] price coverage: "
            f"dropped {len(all_nan_return_cols)}/{px.shape[1]} tickers with no monthly returns; "
            f"sample: {sample}"
        )
    partial_return_cols = sorted(returns.columns[returns.isna().any()].tolist())
    if partial_return_cols:
        sample = ", ".join(partial_return_cols[:20])
        print(
            "  [info] price coverage: "
            f"keeping {len(partial_return_cols)}/{returns.shape[1]} tickers with partial monthly "
            f"history; engine exits or skips them in gap months; sample: {sample}"
        )
    if returns.empty:
        raise RuntimeError("All downloaded yfinance price columns had no monthly returns.")
    returns.attrs["price_diagnostics"] = {
        "tickers_requested": int(len(clean)),
        "tickers_with_close": int(px.shape[1]),
        "tickers_with_complete_returns": int(returns.notna().all().sum()),
        "tickers_with_partial_returns": int(len(partial_return_cols)),
        "tickers_without_close": int(len(no_price)),
        "tickers_no_returns_dropped": int(len(all_nan_return_cols)),
        "invalid_tickers_skipped": int(dropped),
        "tickers_from_cache": int(len(cached_cols)),
        "tickers_refetched_due_to_incomplete_cache": int(len(stale_cached_cols)),
        "tickers_skipped_known_no_close": int(len(coverage_no_close)),
        "used_chart_fallback": bool(used_chart_fallback),
        "failed_batches": failed_batches[:10],
        "empty_batches": empty_batches[:10],
        "chart_failed_batches": chart_failed_batches[:10],
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
