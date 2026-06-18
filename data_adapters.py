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
import json
import multiprocessing as mp
from pathlib import Path
import re
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd
import requests

PRICE_ELIGIBLE_SEC_TYPES = {"SH"}
YFINANCE_TICKER_RE = r"^[A-Z]{1,6}([.\-][A-Z]{1,2})?$"
OPENFIGI_SELECTOR_VERSION = 2
OPENFIGI_METADATA_VERSION = 1
PRICE_COVERAGE_SCHEMA_VERSION = 2
OPENFIGI_US_EQUITY_FILTERS = {
    "currency": "USD",
    "marketSecDes": "Equity",
    "exchCode": "US",
}
OPENFIGI_METADATA_FIELDS = [
    "name",
    "marketSector",
    "marketSecDes",
    "securityType",
    "securityType2",
    "securityDescription",
    "exchCode",
    "currency",
    "figi",
    "compositeFIGI",
    "shareClassFIGI",
]
DEFAULT_FUND_LIKE_TICKERS = {
    "AGG", "ARKK", "BIL", "DIA", "EEM", "EFA", "EMB", "HYG", "IAU", "IEFA",
    "IEMG", "IJH", "IJR", "IVV", "IWB", "IWD", "IWF", "IWM", "IWN", "IWO",
    "LQD", "MDY", "QQQ", "RSP", "SCHA", "SCHF", "SCHX", "SHV", "SLV", "SPY",
    "TIP", "TLT", "USO", "VB", "VBK", "VBR", "VCIT", "VEA", "VGT", "VNQ",
    "VO", "VOE", "VONG", "VOO", "VOT", "VTI", "VTV", "VUG", "VV", "VWO",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
}
FUND_LIKE_ISSUER_RE = re.compile(
    r"\b(?:"
    r"ETF|ETN|EXCHANGE TRADED|ISHARES|SPDR|SELECT SECTOR SPDR|VANGUARD INDEX|"
    r"VANGUARD WORLD|VANGUARD WHITEHALL|VANGUARD SCOTTSDALE|INVESCO QQQ|"
    r"PROSHARES|DIREXION|GLOBAL X|WISDOMTREE|VANECK|VAN ECK|ARK ETF|"
    r"FIRST TR EXCHANGE|FIRST TRUST EXCHANGE|JPMORGAN EXCHANGE TRADED|"
    r"PIMCO ETF|SCHWAB STRATEGIC TR|BLACKROCK ETF|ISHARES TR|ISHARES INC|"
    r"ISHARES U S ETF|SPDR INDEX SHS FDS|SPDR SERIES TRUST"
    r")\b",
    re.IGNORECASE,
)


def _normalise_yfinance_ticker(value) -> str | None:
    if pd.isna(value):
        return None
    ticker = str(value).strip().upper().replace("/", "-")
    return ticker if re.fullmatch(YFINANCE_TICKER_RE, ticker) else None


def _is_yfinance_ticker(value) -> bool:
    return _normalise_yfinance_ticker(value) is not None


def _load_ticker_exclusion_file(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "ticker" not in df.columns:
            raise ValueError(f"ticker exclusion file missing ticker column: {p}")
        raw = df["ticker"]
    else:
        raw = pd.Series(p.read_text(encoding="utf-8").splitlines())
    return {
        t
        for t in raw.map(_normalise_yfinance_ticker).dropna().astype(str)
        if t and not t.startswith("#")
    }


def _openfigi_id_type(identifier: str) -> str:
    ident = str(identifier).strip().upper()
    # CINS identifiers look like CUSIPs but start with a letter for non-US
    # issuers. OpenFIGI returns many foreign-domiciled US-listed 13F names only
    # when queried as ID_CINS, e.g. MDT, ACN, AON, NXPI.
    return "ID_CINS" if ident[:1].isalpha() else "ID_CUSIP"


def _is_us_openfigi_exchange(value) -> bool:
    exch = str(value or "").strip().upper()
    return exch == "US" or (len(exch) == 2 and exch.startswith("U"))


def _is_openfigi_fund_like(ticker: str | None, rec: dict | None) -> bool:
    ticker_norm = _normalise_yfinance_ticker(ticker) if ticker else None
    if ticker_norm in DEFAULT_FUND_LIKE_TICKERS:
        return True
    if not isinstance(rec, dict):
        return False
    text = " ".join(str(rec.get(k, "")) for k in OPENFIGI_METADATA_FIELDS).upper()
    return bool(
        re.search(
            r"\b(?:ETF|ETN|EXCHANGE TRADED|OPEN-END FUND|CLOSED-END FUND|MUTUAL FUND|"
            r"UNIT INVESTMENT TRUST|INDEX FUND)\b",
            text,
        )
    )


def _is_openfigi_common_stock_like(rec: dict | None, *, fund_like: bool) -> bool:
    if fund_like or not isinstance(rec, dict):
        return False
    text = " ".join(str(rec.get(k, "")) for k in ("securityType", "securityType2", "securityDescription")).upper()
    return any(term in text for term in ("COMMON STOCK", "ADR", "REIT", "ORDINARY SHARE", "SHS"))


def _select_openfigi_record(data: list[dict] | None) -> dict | None:
    if not data:
        return None
    candidates = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        ticker = _normalise_yfinance_ticker(rec.get("ticker"))
        if ticker is None:
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
        selected = dict(rec)
        selected["ticker"] = ticker
        candidates.append((score, ticker, selected))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def _select_openfigi_ticker(data: list[dict] | None) -> str | None:
    rec = _select_openfigi_record(data)
    return None if rec is None else _normalise_yfinance_ticker(rec.get("ticker"))


def _openfigi_cache_row(cusip: str, ticker: str | None, rec: dict | None = None) -> dict:
    ticker_norm = _normalise_yfinance_ticker(ticker)
    fund_like = _is_openfigi_fund_like(ticker_norm, rec)
    row = {
        "cusip": str(cusip).strip().upper(),
        "ticker": ticker_norm,
        "id_type": _openfigi_id_type(cusip),
        "selector_version": OPENFIGI_SELECTOR_VERSION,
        "metadata_version": OPENFIGI_METADATA_VERSION if rec else pd.NA,
        "is_fund_like": bool(fund_like),
        "is_common_stock_like": bool(_is_openfigi_common_stock_like(rec, fund_like=fund_like)),
    }
    for field in OPENFIGI_METADATA_FIELDS:
        row[field] = None if rec is None else rec.get(field)
    return row


def _load_openfigi_cache_entries(cache_path: str | Path | None) -> dict[str, dict]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if "cusip" not in df or "ticker" not in df:
        return {}
    entries: dict[str, dict] = {}
    has_selector_version = "selector_version" in df.columns
    for raw in df.to_dict(orient="records"):
        cusip = str(raw.get("cusip", "")).strip().upper()
        if not cusip:
            continue
        ticker = _normalise_yfinance_ticker(raw.get("ticker"))
        selector_version = raw.get("selector_version") if has_selector_version else pd.NA
        is_current_negative = (
            ticker is None
            and has_selector_version
            and pd.notna(selector_version)
            and int(selector_version) >= OPENFIGI_SELECTOR_VERSION
        )
        if ticker is None and not is_current_negative:
            continue
        row = _openfigi_cache_row(cusip, ticker, None)
        for key, value in raw.items():
            row[key] = value
        row["cusip"] = cusip
        row["ticker"] = ticker
        row["id_type"] = row.get("id_type") or _openfigi_id_type(cusip)
        row["selector_version"] = (
            int(row["selector_version"]) if pd.notna(row.get("selector_version")) else OPENFIGI_SELECTOR_VERSION
        )
        entries[cusip] = row
    return entries


def _load_openfigi_cache(cache_path: str | Path | None) -> dict[str, str | None]:
    return {c: row.get("ticker") for c, row in _load_openfigi_cache_entries(cache_path).items()}


def load_openfigi_metadata(cache_path: str | Path | None) -> pd.DataFrame:
    entries = _load_openfigi_cache_entries(cache_path)
    if not entries:
        return pd.DataFrame()
    df = pd.DataFrame(entries.values())
    keep = [
        "cusip",
        "ticker",
        "metadata_version",
        "is_fund_like",
        "is_common_stock_like",
        *OPENFIGI_METADATA_FIELDS,
    ]
    for col in keep:
        if col not in df:
            df[col] = pd.NA
    return df[keep].drop_duplicates("cusip", keep="last")


def _write_openfigi_cache(cache_path: str | Path | None, cache: dict[str, str | None | dict]) -> None:
    if cache_path is None:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c, value in sorted(cache.items()):
        if isinstance(value, dict):
            row = _openfigi_cache_row(c, value.get("ticker"), value)
            row.update(value)
            row["cusip"] = str(c).strip().upper()
            row["ticker"] = _normalise_yfinance_ticker(row.get("ticker"))
            row["id_type"] = row.get("id_type") or _openfigi_id_type(c)
            row["selector_version"] = OPENFIGI_SELECTOR_VERSION
        else:
            row = _openfigi_cache_row(c, value, None)
        rows.append(row)
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


def _yf_download_close_worker(batch: list[str], start: str, end: str, out_path: str, status_path: str) -> None:
    try:
        import yfinance as yf

        close = _yf_download_close(yf, batch, start, end)
        if not close.empty:
            close.to_parquet(out_path)
        Path(status_path).write_text(json.dumps({"ok": True, "empty": bool(close.empty)}), encoding="utf-8")
    except BaseException as exc:
        Path(status_path).write_text(
            json.dumps({"ok": False, "type": type(exc).__name__, "message": str(exc)}),
            encoding="utf-8",
        )


def _yf_download_close_subprocess(batch: list[str], start: str, end: str, timeout_seconds: int) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "close.parquet")
        status_path = str(Path(tmp) / "status.json")
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_yf_download_close_worker,
            args=(batch, start, end, out_path, status_path),
        )
        proc.start()
        proc.join(timeout_seconds)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            raise TimeoutError(
                f"yfinance batch timed out after {timeout_seconds}s: {batch[0]}..{batch[-1]}"
            )
        if not Path(status_path).exists():
            raise RuntimeError(f"yfinance worker exited without status code={proc.exitcode}")
        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
        if not status.get("ok"):
            raise RuntimeError(f"{status.get('type')}: {status.get('message')}")
        if status.get("empty") or not Path(out_path).exists():
            return pd.DataFrame()
        return pd.read_parquet(out_path)


def _yf_download_close_guarded(
    yf,
    batch: list[str],
    start: str,
    end: str,
    timeout_seconds: int | None,
) -> pd.DataFrame:
    if timeout_seconds and timeout_seconds > 0 and getattr(_yf_download_close, "__module__", __name__) == __name__:
        return _yf_download_close_subprocess(batch, start, end, timeout_seconds)
    return _yf_download_close(yf, batch, start, end)


def _looks_rate_limited(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "ratelimit" in text or "rate limit" in text or "too many requests" in text or "429" in text


def _yfinance_probe(yf, start: str, end: str, timeout_seconds: int | None = 60) -> str:
    try:
        close = _yf_download_close_guarded(yf, ["SPY", "IWD", "AAPL"], start, end, timeout_seconds)
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


def _yahoo_chart_download_close_worker(
    batch: list[str],
    start: str,
    end: str,
    max_workers: int,
    out_path: str,
    status_path: str,
) -> None:
    try:
        close = _yahoo_chart_download_close(batch, start, end, max_workers=max_workers)
        if not close.empty:
            close.to_parquet(out_path)
        Path(status_path).write_text(json.dumps({"ok": True, "empty": bool(close.empty)}), encoding="utf-8")
    except BaseException as exc:
        Path(status_path).write_text(
            json.dumps({"ok": False, "type": type(exc).__name__, "message": str(exc)}),
            encoding="utf-8",
        )


def _yahoo_chart_download_close_subprocess(
    batch: list[str],
    start: str,
    end: str,
    *,
    max_workers: int,
    timeout_seconds: int,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "chart_close.parquet")
        status_path = str(Path(tmp) / "chart_status.json")
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_yahoo_chart_download_close_worker,
            args=(batch, start, end, max_workers, out_path, status_path),
        )
        proc.start()
        proc.join(timeout_seconds)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            raise TimeoutError(
                f"Yahoo Chart batch timed out after {timeout_seconds}s: {batch[0]}..{batch[-1]}"
            )
        if not Path(status_path).exists():
            raise RuntimeError(f"Yahoo Chart worker exited without status code={proc.exitcode}")
        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
        if not status.get("ok"):
            raise RuntimeError(f"{status.get('type')}: {status.get('message')}")
        if status.get("empty") or not Path(out_path).exists():
            return pd.DataFrame()
        return pd.read_parquet(out_path)


def _yahoo_chart_download_close_guarded(
    batch: list[str],
    start: str,
    end: str,
    *,
    max_workers: int,
    timeout_seconds: int | None,
) -> pd.DataFrame:
    if (
        timeout_seconds
        and timeout_seconds > 0
        and getattr(_yahoo_chart_download_close, "__module__", __name__) == __name__
    ):
        return _yahoo_chart_download_close_subprocess(
            batch,
            start,
            end,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )
    return _yahoo_chart_download_close(batch, start, end, max_workers=max_workers)


def _yahoo_chart_probe(start: str, end: str, timeout_seconds: int | None = 60) -> str:
    try:
        close = _yahoo_chart_download_close_guarded(
            ["SPY", "IWD", "AAPL"],
            start,
            end,
            max_workers=3,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return f"chart_probe_exception={type(exc).__name__}: {exc}"
    if close.empty:
        return "chart_probe_empty_close"
    non_empty = close.dropna(axis=1, how="all").columns.tolist()
    if not non_empty:
        return "chart_probe_all_nan_close"
    return f"chart_probe_ok={','.join(map(str, non_empty))}"


def _fetch_yahoo_chart_batch_resilient(
    batch: list[str],
    start: str,
    end: str,
    *,
    max_workers: int,
    timeout_seconds: int | None,
    split_threshold: int = 10,
) -> tuple[list[pd.DataFrame], list[str], int]:
    if not batch:
        return [], [], 0
    try:
        close = _yahoo_chart_download_close_guarded(
            batch,
            start,
            end,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )
        return ([close] if not close.empty else []), [], 0
    except TimeoutError as exc:
        if len(batch) == 1:
            return [], [f"{batch[0]}: {exc}"], 1
        chunk_size = max(1, min(split_threshold, len(batch) // 2))
        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        timeouts = 1
        for j in range(0, len(batch), chunk_size):
            child_frames, child_failures, child_timeouts = _fetch_yahoo_chart_batch_resilient(
                batch[j:j + chunk_size],
                start,
                end,
                max_workers=max_workers,
                timeout_seconds=timeout_seconds,
                split_threshold=max(1, split_threshold // 2),
            )
            frames.extend(child_frames)
            failures.extend(child_failures)
            timeouts += child_timeouts
        return frames, failures, timeouts
    except Exception as exc:
        return [], [f"{batch[0]}..{batch[-1]}: {type(exc).__name__}: {exc}"], 0


def _fetch_yahoo_chart_batches(
    tickers: list[str],
    start: str,
    end: str,
    *,
    batch_size: int,
    cache_path: str | Path | None,
    max_workers: int,
    batch_timeout_seconds: int | None = 90,
) -> tuple[list[pd.DataFrame], list[str]]:
    frames: list[pd.DataFrame] = []
    failed_batches: list[str] = []
    n_batches = int(np.ceil(len(tickers) / batch_size)) if tickers else 0
    print(f"  Yahoo Chart fallback: {len(tickers)} tickers in {n_batches} batches")
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_no = i // batch_size + 1
        print(f"    yahoo-chart batch {batch_no}/{n_batches}: {batch[0]}..{batch[-1]}")
        t0 = time.perf_counter()
        batch_frames, batch_failures, batch_timeouts = _fetch_yahoo_chart_batch_resilient(
            batch,
            start,
            end,
            max_workers=max_workers,
            timeout_seconds=batch_timeout_seconds,
        )
        close = (
            pd.concat(batch_frames, axis=1).loc[:, lambda x: ~x.columns.duplicated()].sort_index()
            if batch_frames
            else pd.DataFrame()
        )
        non_nan_close = close.dropna(axis=1, how="all") if not close.empty else close
        if batch_failures:
            failed_batches.extend(batch_failures)
        elapsed = time.perf_counter() - t0
        if non_nan_close.empty:
            failed_batches.append(f"{batch[0]}..{batch[-1]}: empty_close")
            print(
                f"      yahoo-chart batch {batch_no}/{n_batches}: "
                f"0/{len(batch)} close columns in {elapsed:.1f}s"
            )
        else:
            frames.append(non_nan_close)
            _write_price_cache(cache_path, non_nan_close)
            _write_price_coverage(cache_path, non_nan_close.columns, start, end, "fetched", non_nan_close)
            no_close_in_batch = sorted(set(batch) - set(non_nan_close.columns))
            _write_price_coverage(cache_path, no_close_in_batch, start, end, "no_close")
            print(
                f"      yahoo-chart batch {batch_no}/{n_batches}: "
                f"{len(non_nan_close.columns)}/{len(batch)} close columns in {elapsed:.1f}s"
                + (f"; split_timeouts={batch_timeouts}" if batch_timeouts else "")
            )
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
    for col in ("actual_first_close", "actual_last_close"):
        if col not in cov:
            cov[col] = pd.NaT
        cov[col] = pd.to_datetime(cov[col], errors="coerce")
    if "rows" not in cov:
        cov["rows"] = pd.NA
    if "coverage_schema_version" not in cov:
        cov["coverage_schema_version"] = 1
    return cov


def _price_coverage_rows(tickers, start: str, end: str, status: str, px: pd.DataFrame | None = None) -> list[dict]:
    clean = sorted({str(t).strip().upper() for t in tickers if pd.notna(t)})
    if not clean:
        return []
    price_frame = pd.DataFrame()
    if px is not None and not px.empty:
        price_frame = px.copy()
        price_frame.index = pd.to_datetime(price_frame.index)
        price_frame.columns = [str(c).strip().upper() for c in price_frame.columns]
    rows = []
    for ticker in clean:
        actual_first = pd.NaT
        actual_last = pd.NaT
        n_rows = 0
        if ticker in price_frame:
            s = price_frame[ticker].dropna()
            n_rows = int(len(s))
            if n_rows:
                actual_first = s.index.min()
                actual_last = s.index.max()
        rows.append({
            "ticker": ticker,
            "start": pd.Timestamp(start),
            "end": pd.Timestamp(end),
            "status": status,
            "actual_first_close": actual_first,
            "actual_last_close": actual_last,
            "rows": n_rows,
            "coverage_schema_version": PRICE_COVERAGE_SCHEMA_VERSION,
        })
    return rows


def _write_price_coverage(
    cache_path: str | Path | None,
    tickers,
    start: str,
    end: str,
    status: str,
    px: pd.DataFrame | None = None,
) -> None:
    path = _price_coverage_cache_path(cache_path)
    if path is None:
        return
    rows = _price_coverage_rows(tickers, start, end, status, px)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_price_coverage(cache_path)
    new = pd.DataFrame(rows)
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


def _series_has_requested_window_data(s: pd.Series, start: str, end: str) -> bool:
    x = s.dropna()
    if x.empty:
        return False
    s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
    return bool(((x.index >= s_ts) & (x.index <= e_ts)).any())


def _trusted_partial_cache_cols(
    coverage: pd.DataFrame,
    cache_px: pd.DataFrame,
    tickers,
    start: str,
    end: str,
    *,
    max_end_lag_days: int = 10,
) -> set[str]:
    """Previously fetched partial histories that should not be refetched forever.

    This is intentionally conservative: it only trusts cache entries created by
    the schema-v2 writer, whose actual close span/row count matches the parquet,
    and whose latest close reaches the requested end. It is meant for natural
    late-start histories such as IPOs, not stale or truncated data.
    """
    if coverage.empty or cache_px.empty:
        return set()
    clean = sorted({str(t).strip().upper() for t in tickers if pd.notna(t)})
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cov = coverage[
        coverage["ticker"].isin(clean)
        & coverage["status"].eq("fetched")
        & (coverage["start"] <= s)
        & (coverage["end"] >= e)
        & coverage["actual_first_close"].notna()
        & coverage["actual_last_close"].notna()
        & (coverage["coverage_schema_version"].fillna(1).astype(int) >= PRICE_COVERAGE_SCHEMA_VERSION)
    ]
    out: set[str] = set()
    for row in cov.sort_values(["ticker", "start", "end"]).itertuples(index=False):
        ticker = str(row.ticker).strip().upper()
        if ticker not in cache_px:
            continue
        series = cache_px[ticker].dropna()
        if series.empty:
            continue
        if _series_spans_requested_window(series, start, end):
            continue
        actual_first = pd.Timestamp(row.actual_first_close)
        actual_last = pd.Timestamp(row.actual_last_close)
        expected_rows = int(row.rows) if pd.notna(row.rows) else -1
        first_delta = abs((series.index.min() - actual_first).total_seconds())
        last_delta = abs((series.index.max() - actual_last).total_seconds())
        end_lag_days = (e - series.index.max()).days
        if (
            first_delta <= 86400
            and last_delta <= 86400
            and expected_rows == len(series)
            and 0 <= end_lag_days <= max_end_lag_days
            and _series_has_requested_window_data(series, start, end)
        ):
            out.add(ticker)
    return out


def _better_close_history(
    current: pd.Series | None,
    candidate: pd.Series,
    start: str,
    end: str,
) -> bool:
    cand = candidate.dropna()
    if cand.empty:
        return False
    if current is None:
        return True
    cur = current.dropna()
    if cur.empty:
        return True
    if _series_spans_requested_window(cand, start, end) and not _series_spans_requested_window(cur, start, end):
        return True
    return cand.index.min() < cur.index.min() or cand.index.max() > cur.index.max() or len(cand) > len(cur)


def _patch_partial_close_with_chart(
    close: pd.DataFrame,
    start: str,
    end: str,
    *,
    max_workers: int,
    timeout_seconds: int | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if close.empty:
        return close, [], []
    partial = sorted(
        c for c in close.columns
        if not _series_spans_requested_window(close[c], start, end)
    )
    if not partial:
        return close, [], []
    try:
        chart = _yahoo_chart_download_close_guarded(
            partial,
            start,
            end,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return close, [], [f"partial_history_chart_patch_failed: {type(exc).__name__}: {exc}"]
    if chart.empty:
        return close, [], ["partial_history_chart_patch_empty"]
    chart = chart.dropna(axis=1, how="all")
    if chart.empty:
        return close, [], ["partial_history_chart_patch_all_nan"]
    patched = close.copy()
    improved = []
    for ticker in sorted(set(partial).intersection(chart.columns)):
        if _better_close_history(patched[ticker] if ticker in patched else None, chart[ticker], start, end):
            patched = patched.reindex(patched.index.union(chart[ticker].dropna().index))
            patched = patched.drop(columns=[ticker], errors="ignore")
            patched[ticker] = chart[ticker]
            improved.append(ticker)
    return patched.sort_index(), improved, []


def audit_price_cache_coverage(
    cache_path: str | Path | None,
    start: str,
    end: str,
    tickers=None,
) -> pd.DataFrame:
    """Compare coverage metadata with the actual cached close history."""
    clean = None
    if tickers is not None:
        clean = sorted({str(t).strip().upper() for t in tickers if pd.notna(t)})
    px = _load_price_cache(cache_path, start, end)
    cov = _load_price_coverage(cache_path)
    if clean is None:
        clean = sorted(set(px.columns).union(cov["ticker"].tolist() if not cov.empty else []))
    rows = []
    for ticker in clean:
        s = px[ticker].dropna() if ticker in px else pd.Series(dtype=float)
        actual_first = s.index.min() if not s.empty else pd.NaT
        actual_last = s.index.max() if not s.empty else pd.NaT
        spans = _series_spans_requested_window(s, start, end)
        c = cov[cov["ticker"].eq(ticker)] if not cov.empty else pd.DataFrame()
        covering = c[
            (c["start"] <= pd.Timestamp(start))
            & (c["end"] >= pd.Timestamp(end))
        ] if not c.empty else pd.DataFrame()
        status = ",".join(sorted(covering["status"].dropna().astype(str).unique())) if not covering.empty else ""
        false_full_coverage = bool(("fetched" in status.split(",")) and not spans)
        rows.append({
            "ticker": ticker,
            "requested_start": pd.Timestamp(start),
            "requested_end": pd.Timestamp(end),
            "actual_first_close": actual_first,
            "actual_last_close": actual_last,
            "cached_rows": int(len(s)),
            "actual_spans_requested_window": bool(spans),
            "coverage_status": status,
            "false_full_coverage": false_full_coverage,
        })
    return pd.DataFrame(rows)


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
    if not current.empty:
        # New downloads must replace stale shorter histories for the same
        # ticker. Keeping the old duplicate column silently corrupts later
        # cache reads, most visibly for the benchmark series.
        current = current.drop(columns=[c for c in px.columns if c in current.columns], errors="ignore")
        merged = pd.concat([current, px], axis=1)
    else:
        merged = px
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
    require_metadata: bool = False,
) -> dict[str, str]:
    ## FRICTION: this is THE assembly bottleneck. Cache the map to disk and only
    ##           query new CUSIPs. ~25 req/min without a (free) OpenFIGI key.
    url = "https://api.openfigi.com/v3/mapping"
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["X-OPENFIGI-APIKEY"] = api_key
    uniq = sorted({str(c).strip().upper() for c in cusips if pd.notna(c)})
    cache = _load_openfigi_cache_entries(cache_path)
    cached = {c: cache[c].get("ticker") for c in uniq if c in cache}
    mp: dict[str, str] = {c: t for c, t in cached.items() if t is not None}
    todo = []
    for c in uniq:
        if c not in cache:
            todo.append(c)
            continue
        if require_metadata:
            metadata_version = cache[c].get("metadata_version")
            if pd.isna(metadata_version) or int(metadata_version) < OPENFIGI_METADATA_VERSION:
                todo.append(c)
    if cache_path is not None:
        print(f"  OpenFIGI cache: {len(cached)}/{len(uniq)} CUSIPs found at {cache_path}")
        if require_metadata:
            missing_meta = len([c for c in uniq if c in cache and c in todo])
            print(f"  OpenFIGI metadata refresh: {missing_meta} cached CUSIPs missing current metadata")
    n_batches = int(np.ceil(len(todo) / 100)) if todo else 0
    todo_label = "CUSIPs needing mapping/metadata" if require_metadata else "uncached CUSIPs"
    print(f"  OpenFIGI mapping: {len(todo)} {todo_label} in {n_batches} batches")
    for i in range(0, len(todo), 100):
        batch_cusips = todo[i:i + 100]
        batch_no = i // 100 + 1
        print(f"    OpenFIGI batch {batch_no}/{n_batches}: {batch_cusips[0]}..{batch_cusips[-1]}")
        batch = [
            {"idType": _openfigi_id_type(c), "idValue": c, **OPENFIGI_US_EQUITY_FILTERS}
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
            rec = _select_openfigi_record(data)
            ticker = None if rec is None else _normalise_yfinance_ticker(rec.get("ticker"))
            if ticker is not None and rec is not None:
                mp[c] = ticker
                cache[c] = _openfigi_cache_row(c, ticker, rec)
                batch_mapped += 1
            elif data:
                if c not in cache or cache[c].get("ticker") is None:
                    cache[c] = _openfigi_cache_row(c, None, None)
                batch_rejected += 1
            else:
                if c not in cache or cache[c].get("ticker") is None:
                    cache[c] = _openfigi_cache_row(c, None, None)
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
    h = holdings.copy()
    h["cusip"] = h["cusip"].astype(str).str.strip().str.upper()
    if "issuer" not in h:
        h["issuer"] = ""
    mapped = h["cusip"].map(cmap).notna()
    sec_type = h.get("sec_type", pd.Series("SH", index=h.index)).fillna("SH").astype(str).str.upper()
    share_amount_type = h.get("share_amount_type", pd.Series("", index=h.index)).fillna("").astype(str).str.upper()
    price_candidate = sec_type.isin(PRICE_ELIGIBLE_SEC_TYPES) & share_amount_type.ne("PRN")
    total_value = h["value"].sum()
    mapped_value = holdings.loc[mapped, "value"].sum()
    candidate_value = h.loc[price_candidate, "value"].sum()
    candidate_mapped_value = h.loc[price_candidate & mapped, "value"].sum()

    by_sec_type: dict[str, dict[str, float | int]] = {}
    for key, g in h.assign(_mapped=mapped, _sec_type=sec_type).groupby("_sec_type", dropna=False):
        value = g["value"].sum()
        mapped_value_sec = g.loc[g["_mapped"], "value"].sum()
        by_sec_type[str(key)] = {
            "rows": int(len(g)),
            "cusips": int(g["cusip"].nunique()),
            "value": float(value),
            "mapped_value": float(mapped_value_sec),
            "value_coverage": float(mapped_value_sec / value) if value else float("nan"),
        }

    top_unmapped = []
    missing = h.loc[~mapped].copy()
    if not missing.empty:
        missing["_sec_type"] = sec_type.loc[missing.index]
        missing["_share_amount_type"] = share_amount_type.loc[missing.index]
        grouped = (
            missing.groupby(["cusip", "issuer", "_sec_type", "_share_amount_type"], dropna=False)
            .agg(value=("value", "sum"), rows=("value", "size"))
            .sort_values("value", ascending=False)
            .head(25)
            .reset_index()
        )
        top_unmapped = []
        for row in grouped.to_dict(orient="records"):
            top_unmapped.append(
                {
                    "cusip": str(row["cusip"]),
                    "issuer": "" if pd.isna(row["issuer"]) else str(row["issuer"]),
                    "sec_type": str(row["_sec_type"]),
                    "share_amount_type": str(row["_share_amount_type"]),
                    "rows": int(row["rows"]),
                    "value": float(row["value"]),
                    "pct_total_value": float(row["value"] / total_value) if total_value else float("nan"),
                }
            )

    return {
        "rows_total": int(len(holdings)),
        "rows_mapped": int(mapped.sum()),
        "rows_unmapped": int((~mapped).sum()),
        "cusips_total": int(h["cusip"].nunique()),
        "cusips_mapped": int(h.loc[mapped, "cusip"].nunique()),
        "cusips_unmapped": int(h.loc[~mapped, "cusip"].nunique()),
        "value_total": float(total_value),
        "value_mapped": float(mapped_value),
        "value_coverage": float(mapped_value / total_value) if total_value else float("nan"),
        "price_candidate_rows_total": int(price_candidate.sum()),
        "price_candidate_rows_mapped": int((price_candidate & mapped).sum()),
        "price_candidate_cusips_total": int(h.loc[price_candidate, "cusip"].nunique()),
        "price_candidate_cusips_mapped": int(h.loc[price_candidate & mapped, "cusip"].nunique()),
        "price_candidate_value_total": float(candidate_value),
        "price_candidate_value_mapped": float(candidate_mapped_value),
        "price_candidate_value_coverage": (
            float(candidate_mapped_value / candidate_value) if candidate_value else float("nan")
        ),
        "by_sec_type": by_sec_type,
        "top_unmapped_by_value": top_unmapped,
        "drop_reason": "unmapped_cusip",
    }


def map_holdings_to_tickers(
    holdings: pd.DataFrame,
    cmap: dict[str, str],
    *,
    openfigi_metadata: pd.DataFrame | None = None,
    min_value_coverage: float = 0.90,
    strict: bool = False,
) -> pd.DataFrame:
    h = holdings.copy()
    h["cusip"] = h["cusip"].astype(str).str.strip().str.upper()
    h["ticker"] = h["cusip"].map(cmap).astype("string").str.strip()
    h["ticker"] = h["ticker"].mask(h["ticker"].eq(""))
    if openfigi_metadata is not None and not openfigi_metadata.empty:
        meta = openfigi_metadata.copy()
        meta["cusip"] = meta["cusip"].astype(str).str.strip().str.upper()
        meta = meta.drop(columns=["ticker"], errors="ignore")
        add_cols = [c for c in meta.columns if c == "cusip" or c not in h.columns]
        h = h.merge(meta[add_cols].drop_duplicates("cusip", keep="last"), on="cusip", how="left")
    diag = mapping_diagnostics(h, cmap)
    msg = (
        "CUSIP mapping coverage: "
        f"{diag['cusips_mapped']}/{diag['cusips_total']} CUSIPs, "
        f"{diag['rows_mapped']}/{diag['rows_total']} rows, "
        f"{diag['value_coverage']:.1%} value mapped"
    )
    candidate_msg = (
        "CUSIP mapping price-candidate coverage: "
        f"{diag['price_candidate_cusips_mapped']}/{diag['price_candidate_cusips_total']} CUSIPs, "
        f"{diag['price_candidate_rows_mapped']}/{diag['price_candidate_rows_total']} rows, "
        f"{diag['price_candidate_value_coverage']:.1%} long-share value mapped"
    )
    if strict and diag["value_coverage"] < min_value_coverage:
        raise ValueError(msg)
    if diag["rows_unmapped"]:
        print(f"  [warn] {msg}; dropping unmapped_cusip rows")
        print(f"  [warn] {candidate_msg}")
        sample = diag.get("top_unmapped_by_value", [])[:10]
        if sample:
            formatted = ", ".join(
                f"{row['cusip']} {row['issuer']} {row['value']:.0f}"
                for row in sample
            )
            print(f"  [warn] top unmapped CUSIPs by value: {formatted}")
    else:
        print(f"  {msg}")
        print(f"  {candidate_msg}")
    out = h.dropna(subset=["ticker"]).copy()
    out.attrs["mapping_diagnostics"] = diag
    return out


def _fund_like_mask(holdings: pd.DataFrame, extra_fund_tickers: set[str] | None = None) -> pd.Series:
    h = holdings.copy()
    ticker = h.get("ticker", pd.Series("", index=h.index)).astype("string").str.strip().str.upper()
    issuer = h.get("issuer", pd.Series("", index=h.index)).fillna("").astype(str)
    security_text = pd.Series("", index=h.index, dtype="string")
    for col in ("security_type", "securityType", "securityType2", "securityDescription", "marketSecDes"):
        if col in h:
            security_text = security_text.str.cat(h[col].fillna("").astype(str), sep=" ")
    fund_tickers = set(DEFAULT_FUND_LIKE_TICKERS)
    if extra_fund_tickers:
        fund_tickers |= {str(t).strip().upper() for t in extra_fund_tickers if str(t).strip()}
    by_ticker = ticker.isin(fund_tickers)
    by_metadata = (
        h["is_fund_like"].fillna(False).astype(bool)
        if "is_fund_like" in h
        else pd.Series(False, index=h.index)
    )
    by_issuer = issuer.str.contains(FUND_LIKE_ISSUER_RE, na=False)
    by_security_text = security_text.str.contains(
        r"\b(?:ETF|ETN|EXCHANGE TRADED FUND|EXCHANGE TRADED PRODUCT|OPEN-END FUND|CLOSED-END FUND|MUTUAL FUND)\b",
        case=False,
        regex=True,
        na=False,
    )
    return (by_metadata | by_ticker | by_issuer | by_security_text).astype(bool)


def priceable_holdings(
    holdings: pd.DataFrame,
    *,
    exclude_fund_like: bool = False,
    fund_ticker_exclusions_path: str | Path | None = None,
    extra_fund_tickers: set[str] | None = None,
) -> pd.DataFrame:
    """
    Keep holdings that can plausibly be priced as exchange-traded equities by
    yfinance. 13F PRN rows are usually bonds/convertibles and OpenFIGI may map
    them to descriptions like 'ABC 2.25 08/15/28', not stock symbols.

    If exclude_fund_like=True, also drop ETF/ETN/fund-like rows before idea
    generation. This is a research-design filter for equity-only clone tests;
    diagnostics report the dropped exposure so the choice is auditable.
    """
    h = holdings.copy()
    sec_type = h.get("sec_type", pd.Series("SH", index=h.index)).fillna("SH").astype(str).str.upper()
    ticker = h["ticker"].astype("string").str.strip().str.upper()
    keep_sec = sec_type.isin(PRICE_ELIGIBLE_SEC_TYPES)
    keep_ticker = ticker.map(_is_yfinance_ticker).astype(bool)
    extra_from_file = _load_ticker_exclusion_file(fund_ticker_exclusions_path)
    if extra_fund_tickers:
        extra_from_file |= {str(t).strip().upper() for t in extra_fund_tickers if str(t).strip()}
    fund_like = _fund_like_mask(h, extra_from_file) if exclude_fund_like else pd.Series(False, index=h.index)
    out = h.loc[keep_sec & keep_ticker & ~fund_like].copy()
    out["ticker"] = ticker.loc[out.index]
    dropped = int(len(h) - len(out))
    dropped_fund_like = int((keep_sec & keep_ticker & fund_like).sum())
    fund_like_value = float(h.loc[keep_sec & keep_ticker & fund_like, "value"].sum()) if "value" in h else float("nan")
    fund_like_tickers = sorted(ticker.loc[keep_sec & keep_ticker & fund_like].dropna().unique().tolist())[:25]
    if dropped:
        reason = (
            "non-equity sec_type, invalid yfinance ticker, or ETF/ETN/fund-like row"
            if exclude_fund_like
            else "non-equity sec_type or non-yfinance ticker"
        )
        print(
            "  [warn] price input filter: "
            f"dropped {dropped}/{len(h)} mapped rows with {reason}"
            + (f"; fund_like rows dropped {dropped_fund_like}" if exclude_fund_like else "")
        )
    if exclude_fund_like:
        print(
            "  equity-only filter: "
            f"dropped {dropped_fund_like} ETF/ETN/fund-like rows before pricing"
            + (f"; sample: {', '.join(fund_like_tickers[:15])}" if fund_like_tickers else "")
        )
    out.attrs.update(h.attrs)
    out.attrs["price_filter_diagnostics"] = {
        "rows_total": int(len(h)),
        "rows_priceable": int(len(out)),
        "rows_dropped": dropped,
        "rows_fund_like_dropped": dropped_fund_like,
        "value_fund_like_dropped": fund_like_value,
        "tickers_fund_like_dropped_sample": fund_like_tickers,
        "exclude_fund_like": bool(exclude_fund_like),
        "fund_ticker_exclusions_path": str(fund_ticker_exclusions_path) if fund_ticker_exclusions_path else None,
        "drop_reason": (
            "non_equity_or_invalid_yfinance_ticker_or_fund_like"
            if exclude_fund_like
            else "non_equity_or_invalid_yfinance_ticker"
        ),
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
    empty_close_retries: int = 0,
    rate_limit_sleep: int = 60,
    cache_path: str | Path | None = None,
    max_consecutive_empty_batches: int = 3,
    use_chart_fallback: bool = True,
    chart_fallback_workers: int = 8,
    chart_fallback_batch_timeout_seconds: int | None = 90,
    yfinance_batch_timeout_seconds: int | None = 45,
    price_source: str = "auto",
    require_full_window: bool = False,
) -> pd.DataFrame:
    ## FRICTION: yfinance has survivorship gaps (delisted names vanish). For a
    ##           publishable backtest use CRSP via WRDS; yfinance is fine first-pass.
    price_source = str(price_source).strip().lower()
    if price_source not in {"auto", "yfinance", "chart"}:
        raise ValueError("price_source must be one of: auto, yfinance, chart")
    yf = None
    if price_source != "chart":
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
    # Treat cached no_close rows as diagnostics only. Free Yahoo endpoints can
    # omit individual tickers during partial/rate-limited responses, so this is
    # not strong enough evidence to skip future download attempts.
    coverage_no_close: set[str] = set()
    cache_symbol_cols = sorted(set(clean).intersection(cache_px.columns)) if not cache_px.empty else []
    trusted_partial_cols = (
        set()
        if require_full_window
        else _trusted_partial_cache_cols(coverage, cache_px, cache_symbol_cols, start, end)
    )
    cached_cols = sorted(
        t for t in cache_symbol_cols
        if _series_spans_requested_window(cache_px[t], start, end) or t in trusted_partial_cols
    )
    stale_cached_cols = sorted(set(cache_symbol_cols) - set(cached_cols))
    false_coverage_cols = sorted(set(stale_cached_cols).intersection(coverage_fetched))
    frames: list[pd.DataFrame] = []
    if cached_cols:
        cached_px = cache_px[cached_cols].dropna(axis=1, how="all")
        if not cached_px.empty:
            frames.append(cached_px)
        print(f"  yfinance cache: {len(cached_cols)}/{len(clean)} coverage-valid tickers found at {cache_path}")
    if trusted_partial_cols:
        sample = ", ".join(sorted(trusted_partial_cols)[:20])
        print(
            "  price cache: "
            f"reusing {len(trusted_partial_cols)} previously fetched partial-history tickers; sample: {sample}"
        )
    if stale_cached_cols:
        sample = ", ".join(stale_cached_cols[:20])
        print(
            "  [info] yfinance cache: "
            f"refetching {len(stale_cached_cols)} tickers whose cached history does not cover "
            f"{start} to {end}; sample: {sample}"
        )
    if false_coverage_cols:
        sample = ", ".join(false_coverage_cols[:20])
        print(
            "  [warn] yfinance cache coverage metadata disagrees with actual cached history for "
            f"{len(false_coverage_cols)} tickers; refetching; sample: {sample}"
        )
    if coverage_no_close:
        sample = ", ".join(sorted(coverage_no_close)[:20])
        print(f"  yfinance cache: {len(coverage_no_close)} tickers previously fetched with no close data; sample: {sample}")

    todo = sorted(set(clean) - set(cached_cols) - set(coverage_no_close))
    failed_batches: list[str] = []
    empty_batches: list[str] = []
    empty_batch_tickers: list[list[str]] = []
    chart_failed_batches: list[str] = []
    partial_history_patched: list[str] = []
    partial_history_patch_failures: list[str] = []
    used_chart_fallback = False
    consecutive_empty = 0
    n_batches = int(np.ceil(len(todo) / batch_size)) if todo else 0
    print(f"  price download: {len(todo)} uncached / {len(clean)} requested tickers in {n_batches} batches")
    if todo and price_source == "chart":
        chart_frames, chart_failed_batches = _fetch_yahoo_chart_batches(
            todo,
            start,
            end,
            batch_size=batch_size,
            cache_path=cache_path,
            max_workers=chart_fallback_workers,
            batch_timeout_seconds=chart_fallback_batch_timeout_seconds,
        )
        frames.extend(chart_frames)
        used_chart_fallback = True
        todo = []
    stop_yfinance_loop = False
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        batch_no = i // batch_size + 1
        print(f"    yfinance batch {batch_no}/{n_batches}: {batch[0]}..{batch[-1]}")
        for attempt in range(max_retries + 1):
            try:
                close = _yf_download_close_guarded(yf, batch, start, end, yfinance_batch_timeout_seconds)
                if not close.empty:
                    non_nan_close = close.dropna(axis=1, how="all")
                    if non_nan_close.empty:
                        empty_batches.append(f"{batch[0]}..{batch[-1]}: all_nan_close")
                        empty_batch_tickers.append(batch)
                        consecutive_empty += 1
                    else:
                        if use_chart_fallback:
                            patched_close, improved, patch_failures = _patch_partial_close_with_chart(
                                non_nan_close,
                                start,
                                end,
                                max_workers=chart_fallback_workers,
                                timeout_seconds=chart_fallback_batch_timeout_seconds,
                            )
                            if improved:
                                sample = ", ".join(improved[:10])
                                print(
                                    "    [warn] yfinance returned partial history; "
                                    f"patched {len(improved)} tickers via Yahoo Chart; sample: {sample}"
                                )
                                non_nan_close = patched_close.dropna(axis=1, how="all")
                                partial_history_patched.extend(improved)
                                used_chart_fallback = True
                            if patch_failures:
                                partial_history_patch_failures.extend(patch_failures[:5])
                        frames.append(non_nan_close)
                        _write_price_cache(cache_path, non_nan_close)
                        _write_price_coverage(cache_path, non_nan_close.columns, start, end, "fetched", non_nan_close)
                        no_close_in_batch = sorted(set(batch) - set(non_nan_close.columns))
                        _write_price_coverage(cache_path, no_close_in_batch, start, end, "no_close")
                        consecutive_empty = 0
                    break
                last_attempt = attempt >= max_retries
                if attempt < empty_close_retries:
                    wait = min(15 * (attempt + 1), rate_limit_sleep)
                    print(
                        "    [warn] yfinance returned empty Close frame; "
                        f"sleeping {wait}s before retry {attempt + 1}/{empty_close_retries}"
                    )
                    time.sleep(wait)
                    continue
                empty_batches.append(f"{batch[0]}..{batch[-1]}: empty_close")
                empty_batch_tickers.append(batch)
                consecutive_empty += 1
                if price_source == "auto" and use_chart_fallback:
                    remaining = todo[i + batch_size:]
                    fallback_tickers = sorted(set(batch).union(remaining))
                    print(
                        "    [warn] yfinance returned empty Close frame. "
                        f"Switching immediately to Yahoo Chart API fallback for {len(fallback_tickers)} tickers"
                    )
                    chart_frames, chart_failed_batches = _fetch_yahoo_chart_batches(
                        fallback_tickers,
                        start,
                        end,
                        batch_size=batch_size,
                        cache_path=cache_path,
                        max_workers=chart_fallback_workers,
                        batch_timeout_seconds=chart_fallback_batch_timeout_seconds,
                    )
                    frames.extend(chart_frames)
                    used_chart_fallback = True
                    stop_yfinance_loop = True
                break
            except TimeoutError as exc:
                failed_batches.append(f"{batch[0]}..{batch[-1]}: {exc}")
                print(f"    [warn] {exc}")
                if not use_chart_fallback:
                    raise RuntimeError(
                        f"Stopping yfinance download after timed-out batch; "
                        f"recent failed batches: {failed_batches[-3:]}"
                    ) from exc
                remaining = todo[i + batch_size:]
                fallback_tickers = sorted(set(batch).union(remaining))
                chart_probe = _yahoo_chart_probe(start, end, chart_fallback_batch_timeout_seconds)
                if "chart_probe_ok=" not in chart_probe:
                    raise RuntimeError(
                        f"Stopping yfinance download after timed-out batch; "
                        f"Yahoo Chart fallback unavailable: {chart_probe}; "
                        f"recent failed batches: {failed_batches[-3:]}"
                    ) from exc
                print(
                    "    [warn] yfinance batch timed out. "
                    f"Switching to Yahoo Chart API fallback for {len(fallback_tickers)} tickers; {chart_probe}"
                )
                chart_frames, chart_failed_batches = _fetch_yahoo_chart_batches(
                    fallback_tickers,
                    start,
                    end,
                    batch_size=batch_size,
                    cache_path=cache_path,
                    max_workers=chart_fallback_workers,
                    batch_timeout_seconds=chart_fallback_batch_timeout_seconds,
                )
                frames.extend(chart_frames)
                used_chart_fallback = True
                stop_yfinance_loop = True
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
        if stop_yfinance_loop:
            break
        if consecutive_empty >= max_consecutive_empty_batches:
            probe = _yfinance_probe(yf, start, end, yfinance_batch_timeout_seconds)
            if "probe_ok=" not in probe:
                if not use_chart_fallback:
                    raise RuntimeError(
                        "Stopping yfinance download after "
                        f"{consecutive_empty} consecutive empty batches; {probe}; "
                        f"recent empty batches: {empty_batches[-max_consecutive_empty_batches:]}"
                    )
                chart_probe = _yahoo_chart_probe(start, end, chart_fallback_batch_timeout_seconds)
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
                    batch_timeout_seconds=chart_fallback_batch_timeout_seconds,
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
        probe = _yfinance_probe(yf, start, end, yfinance_batch_timeout_seconds)
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
        "tickers_from_trusted_partial_cache": int(len(trusted_partial_cols)),
        "tickers_refetched_due_to_incomplete_cache": int(len(stale_cached_cols)),
        "tickers_refetched_due_to_false_coverage": int(len(false_coverage_cols)),
        "false_coverage_tickers": false_coverage_cols[:50],
        "tickers_skipped_known_no_close": int(len(coverage_no_close)),
        "used_chart_fallback": bool(used_chart_fallback),
        "price_source": price_source,
        "empty_close_retries": int(empty_close_retries),
        "yfinance_batch_timeout_seconds": (
            int(yfinance_batch_timeout_seconds) if yfinance_batch_timeout_seconds else None
        ),
        "chart_fallback_batch_timeout_seconds": (
            int(chart_fallback_batch_timeout_seconds) if chart_fallback_batch_timeout_seconds else None
        ),
        "failed_batches": failed_batches[:10],
        "empty_batches": empty_batches[:10],
        "chart_failed_batches": chart_failed_batches[:10],
        "partial_history_patched_tickers": sorted(set(partial_history_patched))[:50],
        "partial_history_patch_failures": partial_history_patch_failures[:10],
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
