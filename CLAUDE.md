# AutoDSC — Canonical Naming, Color & Style Conventions

This file is the **single source of truth** for how methods and datasets are
named, colored, ordered, and grouped across **every** artifact in this project:
the paper (`AutoDSC_ICDM_26/paper.tex`), all tables, and all charts
(`results/analysis/`). When adding or editing any figure/table/text, use
these exact names, colors, markers, and ordering. The machine-readable version
lives in `results/analysis/style.py` — import it; do not hard-code.

---

## Methods

Canonical order (use this order in every table and legend):

| # | Canonical name | Group | Result-file prefix | Color (hex) | Marker | Notes |
|---|----------------|-------|--------------------|-------------|--------|-------|
| 1 | **AutoDSC**       | AutoML        | `AutoDSC`                  | `#D55E00` (vermilion-red) | `o` | **Ours** — always drawn thick (lw≈3.4), emphasized |
| 2 | **ConfStream**    | AutoML        | `ConfStream`               | `#0072B2` (blue)   | `s` | full name |
| 3 | **OACE**          | AutoML        | `OnlineAutoClustEnsemble`  | `#E69F00` (orange) | `^` | short for OnlineAutoClust; expand once on first mention |
| 4 | **MOStream**      | Rule-based    | `MOStream`                 | `#CC79A7` (purple) | `h` | full name |
| 5 | **RandomSearch**  | Baselines     | `RandomSearchStream`       | `#009E73` (green)  | `D` | full name; **NOT** AutoML |
| 6 | **Default(Best)** | Baselines     | *(computed, see below)*    | `#555555` (dark grey) | `X` | OnlineAutoClust default-baseline |

**Groups (for table headers / text):** `AutoML` = {AutoDSC, ConfStream, OACE};
`Rule-based` = {MOStream}; `Baselines` = {RandomSearch, Default(Best)}.

**Default(Best)** = the OnlineAutoClust *default-baseline* `\cite{el2023onlineautoclust}`:
per dataset, run the four River fixed algorithms at their **default**
hyperparameters and report the **best-performing** one (selected by ARI; report
that same model's ARI and CMM).

### Single fixed algorithms
Full names: **CluStream, DenStream, DBSTREAM, StreamKMeans** (note casing:
`DBSTREAM` all-caps, `StreamKMeans` camel). They appear individually **only** in
the population-evolution chart; everywhere else they are summarized by
Default(Best). Evolution-chart fill colors (no overlap with the 6 methods):
CluStream `#0072B2`, DenStream `#E69F00`, DBSTREAM `#009E73`, StreamKMeans `#CC79A7`.

---

## Datasets

Canonical order: **Real first, then Synthetic.** Group labels are exactly
**"Real"** and **"Synthetic"**.

| Result-file token | Display name | Group |
|-------------------|--------------|-------|
| `real_insects`            | Insects     | Real |
| `real_kdd99`              | KDD99       | Real |
| `real_pendigits`          | PenDigits   | Real |
| `real_shuttle`            | Shuttle     | Real |
| `real_vehicle`            | Vehicle     | Real |
| `synth_blobs_expanding`   | Expanding   | Synthetic |
| `synth_blobs_gradual`     | Gradual     | Synthetic |
| `synth_blobs_merge_split` | Merge-Split | Synthetic |
| `synth_blobs_sudden`      | Sudden      | Synthetic |
| `synth_rbf_fast`          | Fast-RBF    | Synthetic |

---

## Metrics
Quality: **ARI** (primary, chance-adjusted) and **CMM** (stream-aware). Both
↑ higher-is-better, reported to 3 decimals. Efficiency: **throughput** (ips, ↑),
**latency** (µs/point, ↓), **memory** (MB, ↓).

## Chart style (all figures)
- Font: DejaVu Sans, bold tick labels and axis labels.
- Palette: Okabe-Ito colorblind-safe (above). AutoDSC always vermilion-red + thick.
- One marker per method (above), consistent everywhere.
- No top/right spines; no chart title (caption carries the description).
- Vector PDF + 300-dpi PNG output to `results/analysis/charts/`.
- Legends: bold, single row on top where they fit.

## Rule
Charts and the main results table show the **6 canonical methods**. Do not
reintroduce the 4 single algorithms as separate columns/series outside the
evolution chart. Import names/colors/markers/order from `style.py` — never
re-hard-code them in individual scripts.
