# Point-in-time SEC and 13F rules

Read this file for SEC ingestion, date parsing, holdings versions, amendments,
mapping, eligibility, or 13F-universe tasks.

## Availability

- Decisions use `FILING_DATE` availability, not only `PERIODOFREPORT`.
- Use the exact filing/amendment version available as of the decision timestamp,
  not the latest revised record.
- Treat `13F-HR/A` as point-in-time events.
- Confidential-treatment positions become available only when disclosed.
- Do not infer intra-quarter trades or use future manager survival.

## Parsing and identifiers

- Parse SEC dates through one tested helper with explicit formats.
- Never combine a hard-coded format and `errors="coerce"` without asserting and
  reporting parse success, failures, and representative samples.
- Keep CUSIP-to-ticker mapping separate and auditable; use as-of mapping where
  possible.
- Mark stale, unmapped, delisted, merged, renamed, and non-common instruments.
- Enforce common-stock eligibility before portfolio construction. Yahoo accepting
  a ticker does not make an instrument eligible.

## Coverage reporting

Report filings used, holdings mapped, mapped market value, dropped securities,
drop reasons, all-value coverage, price-candidate coverage, and top unmapped
CUSIPs by value. Material residual unmapped value is a research-validity risk.

ETF/ETN/fund-like holdings are excluded by default for equity-only idea runs.
For deliberate hedge-fund beta-allocation research, enable them explicitly,
label the run ETF-inclusive, and report ETF exposure separately.
