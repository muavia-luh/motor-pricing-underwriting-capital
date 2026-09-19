# Data source

The `freMTPL2freq` / `freMTPL2sev` CSVs are **not committed** (third-party academic
data). Regenerate them with:

```bash
python3 scripts/00_get_data.py
```

That clones the CASdatasets R package and parses the `.rda` files with the pure-Python
reader in `scripts/rda_reader.py`. Alternatively the same data is on OpenML
(dataset IDs 41214 and 41215).

- **freMTPL2freq** — 677,991 policy records representing approximately 358,343
  exposure-years: claim count, exposure, and rating factors.
- **freMTPL2sev** — 26,444 individual claim amounts, linked by `IDpol`.

**Citation:** Christophe Dutang & Arthur Charpentier (2025), *CASdatasets*, R package
v1.2-1, DOI 10.57745/P0KHAG; distributed with *Computational Actuarial Science with R*
(A. Charpentier, ed., CRC Press). The fetch script pins commit
`227fb56b8734bdb7c0327a41180e01d2ddaeaf26`.

**Claim counts:** the historical freMTPL2 issue (claims on policy IDs ≤ 24500 lacking a
severity counterpart) is resolved in v1.2. The counts are:

- reported claims (uncapped sum of `ClaimNb`): **26,444** — equal to the number of
  severity records, so frequency and severity are measured on the same claims;
- claims after capping `ClaimNb` at 4 per policy: **26,405**;
- claims removed by the cap: **39**.

We also cap `Exposure` at 1. `reporting.py` re-checks the reported-vs-severity
reconciliation at run time.
