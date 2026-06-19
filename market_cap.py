"""Historical market-cap inputs for manager-held market-cap benchmarks.

The engine consumes a long table with month_end, ticker, market_cap, and
available_date.  The optional Yahoo builder is a free-data research proxy: it
multiplies unadjusted month-end close by Yahoo's historical shares-outstanding
series.  Yahoo can revise that history, so it is not a strict vendor PIT source.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


SOURCE_YAHOO_SHARES_PROXY = "yahoo_shares_proxy"
MARKET_CAP_METHOD_VERSION = 2


def _normalise_ticker(value) -> str | None:
    if pd.isna(value):
        return None
    ticker = str(value).strip().upper().replace(".", "-").replace("/", "-")
    return ticker or None


def _parse_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value) if pd.notna(value) else False


def _future_split_factors(dates: pd.Series, split_events: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    """Undo Yahoo's future-split adjustment to express close in as-of units."""
    if not split_events:
        return pd.Series(1.0, index=dates.index, dtype=float)
    clean = [(pd.Timestamp(date), float(ratio)) for date, ratio in split_events if ratio > 0]
    return dates.map(lambda date: float(np.prod([ratio for split_date, ratio in clean if split_date > date])))


def _request_json(url: str, *, params: dict, headers: dict, timeout: int, attempts: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code in {404, 410}:
                return {}
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Yahoo request failed after {attempts} attempts: {last_error}")


def load_market_cap_table(path: str | Path) -> pd.DataFrame:
    """Load and validate a long historical market-cap table.

    External PIT data should provide ``available_date`` explicitly.  Generated
    Yahoo proxy rows use the month-end price observation date as availability.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"market-cap file not found: {p}")
    if p.suffix.lower() == ".parquet":
        raw = pd.read_parquet(p)
    elif p.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(p)
    else:
        raw = pd.read_csv(p)
    cols = {str(c).strip().lower().replace(" ", "_"): c for c in raw.columns}

    def pick(names: list[str], *, required: bool = True):
        for name in names:
            if name in cols:
                return cols[name]
        if required:
            raise ValueError(f"market-cap file {p} missing one of columns: {names}")
        return None

    month_col = pick(["month_end", "date", "as_of_date", "asof_date"])
    ticker_col = pick(["ticker", "symbol"])
    cap_col = pick(["market_cap", "marketcap", "mkt_cap", "mcap"])
    available_col = pick(["available_date", "observed_date", "published_date"], required=False)
    source_col = pick(["source"], required=False)
    strict_col = pick(["strict_pit"], required=False)
    out = pd.DataFrame({
        "month_end": pd.to_datetime(raw[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp("M"),
        "ticker": raw[ticker_col].map(_normalise_ticker),
        "market_cap": pd.to_numeric(raw[cap_col], errors="coerce"),
    })
    out["available_date"] = (
        pd.to_datetime(raw[available_col], errors="coerce")
        if available_col is not None else out["month_end"]
    )
    out["source"] = raw[source_col].astype(str) if source_col is not None else "external"
    out["strict_pit"] = raw[strict_col].map(_parse_bool) if strict_col is not None else False
    out = out.dropna(subset=["month_end", "ticker", "market_cap", "available_date"])
    out = out[np.isfinite(out["market_cap"]) & (out["market_cap"] > 0)]
    if out.empty:
        raise ValueError(f"market-cap file has no usable positive point-in-time rows: {p}")
    out = (out.sort_values(["ticker", "month_end", "available_date"])
              .drop_duplicates(["ticker", "month_end"], keep="last"))
    return out.reset_index(drop=True)


def market_caps_by_month(
    table: pd.DataFrame,
    months,
    *,
    max_stale_days: int = 45,
) -> dict[pd.Timestamp, pd.Series]:
    """Return month -> raw market caps, using only values available by month."""
    required = {"month_end", "ticker", "market_cap", "available_date"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"market-cap table missing columns: {sorted(missing)}")
    tbl = table.copy()
    tbl["month_end"] = pd.to_datetime(tbl["month_end"]).dt.to_period("M").dt.to_timestamp("M")
    tbl["available_date"] = pd.to_datetime(tbl["available_date"])
    tbl = tbl.sort_values(["ticker", "month_end", "available_date"])
    out: dict[pd.Timestamp, pd.Series] = {}
    for raw_month in pd.Index(pd.to_datetime(months)).sort_values().unique():
        month = pd.Timestamp(raw_month).to_period("M").to_timestamp("M")
        eligible = tbl[(tbl["month_end"] <= month) & (tbl["available_date"] <= month)]
        if eligible.empty:
            out[pd.Timestamp(raw_month)] = pd.Series(dtype=float)
            continue
        latest = eligible.groupby("ticker", sort=True, as_index=False).tail(1).copy()
        if max_stale_days is not None:
            latest = latest[(month - latest["month_end"]).dt.days <= int(max_stale_days)]
        caps = latest.set_index("ticker")["market_cap"].astype(float)
        out[pd.Timestamp(raw_month)] = caps[caps > 0].sort_index()
    return out


def _yahoo_market_cap_one(
    ticker: str,
    start: str,
    end: str,
    *,
    request_timeout: int = 20,
    max_shares_stale_days: int = 550,
) -> pd.DataFrame:
    symbol = _normalise_ticker(ticker)
    if symbol is None:
        return pd.DataFrame()
    start_utc = pd.Timestamp(start, tz="UTC")
    end_utc = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    period1, period2 = int(start_utc.timestamp()), int(end_utc.timestamp())
    headers = {"User-Agent": "Mozilla/5.0"}

    chart_payload = _request_json(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history,splits",
            "includeAdjustedClose": "false",
        },
        headers=headers,
        timeout=request_timeout,
    )
    chart_result = ((chart_payload.get("chart") or {}).get("result") or [])
    if not chart_result:
        return pd.DataFrame()
    result = chart_result[0]
    timestamps = result.get("timestamp") or []
    closes = ((((result.get("indicators") or {}).get("quote") or [{}])[0]) or {}).get("close") or []
    if not timestamps or not closes:
        return pd.DataFrame()
    close = pd.Series(
        closes,
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
        dtype=float,
    ).dropna().sort_index()
    if close.empty:
        return pd.DataFrame()
    split_events: list[tuple[pd.Timestamp, float]] = []
    for event in (((result.get("events") or {}).get("splits") or {}).values()):
        try:
            split_date = pd.to_datetime(event["date"], unit="s", utc=True).tz_convert(None)
            numerator = float(event.get("numerator"))
            denominator = float(event.get("denominator"))
            ratio = numerator / denominator
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if ratio > 0:
            split_events.append((split_date, ratio))

    shares_payload = _request_json(
        f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}",
        params={"symbol": symbol, "period1": period1, "period2": period2},
        headers=headers,
        timeout=request_timeout,
    )
    results = ((shares_payload.get("timeseries") or {}).get("result") or [])
    share_block = next((item for item in results if item.get("shares_out")), None)
    if not share_block:
        return pd.DataFrame()
    share_series = pd.Series(
        share_block["shares_out"],
        index=pd.to_datetime(share_block.get("timestamp", []), unit="s", utc=True).tz_convert(None),
        dtype=float,
    ).dropna().sort_index()
    share_series = share_series[share_series > 0]
    if share_series.empty:
        return pd.DataFrame()

    close_frame = close.rename("close").to_frame()
    close_frame["price_observed_date"] = close_frame.index
    monthly = close_frame.resample("ME").last().dropna(subset=["close"]).reset_index(names="month_end")
    monthly["split_adjustment"] = _future_split_factors(monthly["price_observed_date"], split_events)
    monthly["asof_unadjusted_close"] = monthly["close"] * monthly["split_adjustment"]
    shares_frame = share_series.rename("shares_outstanding").to_frame().reset_index(names="shares_observed_date")
    merged = pd.merge_asof(
        monthly.sort_values("price_observed_date"),
        shares_frame.sort_values("shares_observed_date"),
        left_on="price_observed_date",
        right_on="shares_observed_date",
        direction="backward",
        tolerance=pd.Timedelta(days=int(max_shares_stale_days)),
    ).dropna(subset=["shares_outstanding"])
    if merged.empty:
        return pd.DataFrame()
    merged["ticker"] = symbol
    merged["shares_split_adjustment"] = _future_split_factors(
        merged["shares_observed_date"],
        split_events,
    )
    merged["split_adjusted_shares"] = merged["shares_outstanding"] * merged["shares_split_adjustment"]
    # Yahoo historical Close is split-adjusted to a common share basis. Align
    # the shares series to the same basis, including split-transition months.
    merged["market_cap"] = merged["close"] * merged["split_adjusted_shares"]
    merged["available_date"] = merged[["price_observed_date", "shares_observed_date"]].max(axis=1)
    merged["source"] = SOURCE_YAHOO_SHARES_PROXY
    merged["strict_pit"] = False
    merged["method_version"] = MARKET_CAP_METHOD_VERSION
    keep = [
        "month_end", "ticker", "market_cap", "available_date", "source", "strict_pit",
        "close", "split_adjustment", "asof_unadjusted_close", "shares_outstanding",
        "shares_split_adjustment", "split_adjusted_shares",
        "price_observed_date", "shares_observed_date",
        "method_version",
    ]
    return merged[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=["market_cap"])


def _coverage_path(cache_path: str | Path) -> Path:
    p = Path(cache_path)
    return p.with_name(f"{p.stem}_coverage.parquet")


def fetch_market_cap_history(
    tickers,
    start: str,
    end: str,
    *,
    cache_path: str | Path,
    batch_size: int = 25,
    max_workers: int = 6,
    request_timeout: int = 20,
    max_shares_stale_days: int = 550,
    retry_error_after_hours: int = 24,
    retry_no_data_after_days: int = 30,
) -> pd.DataFrame:
    """Incrementally build a Yahoo research-proxy market-cap cache."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wanted = sorted({_normalise_ticker(t) for t in tickers} - {None})
    current = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    cov_path = _coverage_path(path)
    coverage = pd.read_parquet(cov_path) if cov_path.exists() else pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    completed: set[str] = set()
    if not coverage.empty:
        coverage = coverage.copy()
        coverage["attempted_at"] = pd.to_datetime(coverage["attempted_at"], errors="coerce")
        latest = coverage.sort_values(["ticker", "attempted_at"]).groupby("ticker", as_index=False).tail(1)
        for row in latest.itertuples(index=False):
            cached_version = getattr(row, "method_version", 0)
            if pd.isna(cached_version) or int(cached_version) != MARKET_CAP_METHOD_VERSION:
                continue
            covers = pd.Timestamp(row.start) <= pd.Timestamp(start) and pd.Timestamp(row.end) >= pd.Timestamp(end)
            age = now - pd.Timestamp(row.attempted_at)
            fresh_error = str(row.status) == "error" and age < pd.Timedelta(hours=retry_error_after_hours)
            fresh_no_data = str(row.status) == "no_data" and age < pd.Timedelta(days=retry_no_data_after_days)
            if covers and (str(row.status) == "ok" or fresh_no_data or fresh_error):
                completed.add(str(row.ticker))
    todo = [ticker for ticker in wanted if ticker not in completed]
    print(f"  market-cap cache: {len(wanted) - len(todo)}/{len(wanted)} ticker requests covered at {path}")
    print(f"  market-cap download: {len(todo)} tickers in {(len(todo) + batch_size - 1) // batch_size} batches")
    new_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    for offset in range(0, len(todo), batch_size):
        batch = todo[offset:offset + batch_size]
        batch_no = offset // batch_size + 1
        total_batches = (len(todo) + batch_size - 1) // batch_size
        t0 = time.perf_counter()
        print(f"    market-cap batch {batch_no}/{total_batches}: {batch[0]}..{batch[-1]}")
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(batch)))) as pool:
            futures = {
                pool.submit(
                    _yahoo_market_cap_one,
                    ticker,
                    start,
                    end,
                    request_timeout=request_timeout,
                    max_shares_stale_days=max_shares_stale_days,
                ): ticker
                for ticker in batch
            }
            for future in as_completed(futures):
                ticker = futures[future]
                attempted_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
                try:
                    frame = future.result()
                    status = "no_data" if frame.empty else "ok"
                    if not frame.empty:
                        new_frames.append(frame)
                except Exception as exc:
                    status = "error"
                    print(f"      [warn] market-cap {ticker}: {type(exc).__name__}: {exc}")
                coverage_rows.append({
                    "ticker": ticker,
                    "start": pd.Timestamp(start),
                    "end": pd.Timestamp(end),
                    "status": status,
                    "attempted_at": attempted_at,
                    "method_version": MARKET_CAP_METHOD_VERSION,
                })
        if new_frames:
            fresh = pd.concat(new_frames, ignore_index=True)
            current = fresh if current.empty else pd.concat([current, fresh], ignore_index=True)
            current = (current.sort_values(["ticker", "month_end", "available_date"])
                              .drop_duplicates(["ticker", "month_end"], keep="last"))
            current.to_parquet(path, index=False)
            new_frames.clear()
        cov_new = pd.DataFrame(coverage_rows)
        cov_out = cov_new if coverage.empty else pd.concat([coverage, cov_new], ignore_index=True)
        cov_out = (cov_out.sort_values(["ticker", "attempted_at"])
                          .drop_duplicates(["ticker", "start", "end"], keep="last"))
        cov_out.to_parquet(cov_path, index=False)
        coverage = cov_out
        coverage_rows.clear()
        ok_count = int((coverage["status"] == "ok").sum())
        print(f"      checkpointed in {time.perf_counter() - t0:.1f}s; total ok requests={ok_count}")
    if not path.exists():
        raise RuntimeError("No market-cap data downloaded; cannot build manager_held_mcap benchmark")
    return load_market_cap_table(path)
