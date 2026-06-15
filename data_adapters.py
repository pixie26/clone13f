"""
Data adapters — the ONLY part that touches the network. Run on your infra
(open internet); everything in engine.py is pure pandas and needs none of this.

Each function returns the standardized frames engine.py expects. All sources here
are public + free. Frictions are flagged inline with  ## FRICTION.

Deps:  pip install edgartools yfinance pandas-datareader requests
NOTE:  edgartools' attribute surface drifts across versions — the 13F parser
       below is written defensively; verify column names against YOUR installed
       version once (one print of f.infotable.columns) and adjust the mapping.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import requests


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
def cusip_to_ticker(cusips, api_key: str | None = None) -> dict[str, str]:
    ## FRICTION: this is THE assembly bottleneck. Cache the map to disk and only
    ##           query new CUSIPs. ~25 req/min without a (free) OpenFIGI key.
    url = "https://api.openfigi.com/v3/mapping"
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["X-OPENFIGI-APIKEY"] = api_key
    uniq = sorted(set(map(str, cusips)))
    mp: dict[str, str] = {}
    for i in range(0, len(uniq), 100):
        batch = [{"idType": "ID_CUSIP", "idValue": c} for c in uniq[i:i + 100]]
        r = requests.post(url, json=batch, headers=hdr, timeout=30)
        if r.status_code == 429:
            time.sleep(6); r = requests.post(url, json=batch, headers=hdr, timeout=30)
        for c, res in zip(uniq[i:i + 100], r.json()):
            data = res.get("data") if isinstance(res, dict) else None
            if data:
                mp[c] = data[0].get("ticker")
        time.sleep(2.5 if not api_key else 0.3)
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
    h["ticker"] = h["cusip"].map(cmap)
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


# --------------------------------------------------------------------------- #
# 3) Monthly stock returns  (yfinance free; prefer CRSP if you have WRDS)
# --------------------------------------------------------------------------- #
def fetch_prices(tickers, start: str, end: str) -> pd.DataFrame:
    ## FRICTION: yfinance has survivorship gaps (delisted names vanish). For a
    ##           publishable backtest use CRSP via WRDS; yfinance is fine first-pass.
    import yfinance as yf
    px = yf.download(list(set(tickers)), start=start, end=end,
                     auto_adjust=True, progress=False)["Close"]
    monthly = px.resample("ME").last()
    return monthly.pct_change(fill_method=None).iloc[1:]


# --------------------------------------------------------------------------- #
# 4) Fama-French 5 + Momentum  (Ken French library, public + free)
# --------------------------------------------------------------------------- #
def fetch_factors(start: str, end: str) -> pd.DataFrame:
    import pandas_datareader.data as web
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start, end)[0] / 100.0
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench", start, end)[0] / 100.0
    f = ff5.rename(columns={"Mkt-RF": "MKT"}).join(mom.rename(columns={mom.columns[0]: "MOM"}))
    f.index = f.index.to_timestamp("M") + pd.offsets.MonthEnd(0)
    return f[["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]]


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
