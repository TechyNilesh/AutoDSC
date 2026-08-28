# AutoDSC: Automated Algorithm Selection and Tuning for Non-Stationary Data Stream Clustering

This repository contains the implementation of **AutoDSC** (**Auto**mated **D**ata **S**tream **C**lustering), an AutoML framework that continuously searches the joint algorithm-and-hyperparameter space for real-time stream clustering.

**Repository:** <https://github.com/TechyNilesh/AutoDSC>

Data stream clustering must adapt to evolving distributions without ground-truth labels, so no single algorithm or hyperparameter configuration remains effective throughout a stream. AutoDSC addresses this by employing **Population-Guided Stochastic Search Optimisation (PGSSO)**, which combines a recency-weighted elite archive with adaptive exploration. A composite of three internal cluster-validity indices (Silhouette, Davies–Bouldin, Calinski–Harabasz) enables label-free configuration selection. Adaptation is continuous and implicit: AutoDSC uses **no explicit drift detector** and never resets its population. Across five real-world and five synthetic streams, AutoDSC attains the best mean rank on both Adjusted Rand Index (ARI) and Cluster Mapping Measure (CMM) while remaining computationally competitive with stream-clustering AutoML baselines.

## Project Structure

```
AutoDSC/
├── src/
│   ├── autodsc/
│   │   └── AutoDSC.py            # Main AutoDSC framework (AutoDSC + PGSSO)
│   ├── baselines/                # Competitor implementations
│   │   ├── ConfStream/ConfStream.py
│   │   ├── OnlineAutoClust/OnlineAutoClust.py
│   │   └── RandomSearchStream/RandomSearchStream.py
│   └── stream_clustering/        # Fast Rust implementation
│       ├── Cargo.toml
│       ├── pyproject.toml
│       └── src/
│           ├── lib.rs
│           ├── clustream.rs
│           ├── dbstream.rs
│           ├── denstream.rs
│           ├── kmeans.rs
│           ├── streamkmeans.rs
│           └── ...
├── datasets/                      # Benchmark datasets + synthetic generators
│   ├── real_insects.csv
│   ├── real_kdd99.csv
│   ├── real_pendigits.csv
│   ├── real_shuttle.csv
│   ├── real_vehicle.csv
│   ├── synth_blobs_expanding.csv
│   ├── synth_blobs_gradual.csv
│   ├── synth_blobs_merge_split.csv
│   ├── synth_blobs_sudden.csv
│   ├── synth_rbf_fast.csv
│   ├── generate_synth_blobs.py
│   └── generate_synth_rbf.py
├── experiments/                   # Experiment runners
│   ├── run_scripts.py            # Main experiment runner
│   ├── run_autodsc_ablation.py   # Ablation study runner
│   ├── run_sensitivity_analysis.py
│   ├── run_alpha_trace.py        # Exploration-rate trace
│   └── scripts/                  # Individual algorithm runners
│       ├── run_autodsc.py
│       ├── run_clustream.py
│       ├── run_confstream.py
│       ├── run_dbstream.py
│       ├── run_denstream.py
│       ├── run_kmeans.py
│       ├── run_mostream.py
│       ├── run_onlineautoclust.py
│       ├── run_randomsearch.py
│       ├── run_streamkmeans.py
│       ├── ablation/
│       │   └── AutoDSC_ablation.py   # Leave-one-out AutoDSC variants
│       └── utils/
│           ├── cmm.py
│           ├── metrics.py
│           └── purityscore.py
├── results/                       # Experiment results and analysis
│   ├── benchmark/                # Main benchmark runs      (JSON, not tracked)
│   ├── ablation/                 # Ablation runs            (JSON, not tracked)
│   ├── sensitivity/              # Sensitivity-analysis runs (JSON, not tracked)
│   └── analysis/
│       ├── plot_*.py             # Chart scripts
│       ├── run_extra_analysis.py # Builds the extra/ intermediate JSON
│       ├── make_supp_table.py    # Supplementary tables
│       ├── style.py              # Canonical colors, markers, method order
│       ├── extra/                # Intermediate JSON        (not tracked)
│       └── charts/               # Generated PDF/PNG figures
│           ├── ablation/
│           ├── alpha/
│           ├── benchmark/
│           ├── cd/
│           ├── dimscaling/
│           ├── efficiency/
│           ├── evolution/
│           ├── overhead/
│           ├── recovery/
│           ├── sensitivity/
│           └── validity/
└── requirements.txt               # Python dependencies
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- Rust (for compiling the fast Rust implementation)
- pip

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TechyNilesh/AutoDSC.git
   cd AutoDSC
   ```
2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Compile Rust implementation (for faster performance):**

   ```bash
   cd src/stream_clustering
   pip install maturin
   maturin develop --release
   cd ../..
   ```

## Running Experiments

### Two Implementation Versions

The project provides two identical implementations with the same functionality — the only difference is the import source:

1. **Python with River Library (Standard):** Uses the River library for stream clustering algorithms
2. **Rust Implementation (Fast):** Uses compiled Rust code for improved performance

Both versions provide the same clustering algorithms and functionality. To switch between versions, simply change the import statements:

#### River Library Version (Standard):

```python
from river.cluster import CluStream, DenStream, STREAMKMeans
```

#### Rust Implementation Version (Fast):

```python
from stream_clustering import CluStream, DenStream, DBStream, KMeans, StreamKMeans
```

The Rust version requires compilation but offers significantly better performance, especially on large datasets.

### Basic Usage

#### Using the Main Runner Script

The `experiments/run_scripts.py` provides multiple execution modes:

```bash
# Run all models on a specific dataset
python experiments/run_scripts.py --mode dataset --name real_insects

# Run a specific model on all datasets
python experiments/run_scripts.py --mode model --name autodsc

# Run all models on all datasets
python experiments/run_scripts.py --mode all

# Run comprehensive evaluation with multiple runs
python experiments/run_scripts.py --mode random
```

#### Benchmark Datasets

| Display name | Result-file token          | Group     |
| ------------ | -------------------------- | --------- |
| Insects      | `real_insects`             | Real      |
| KDD99        | `real_kdd99`               | Real      |
| PenDigits    | `real_pendigits`           | Real      |
| Shuttle      | `real_shuttle`             | Real      |
| Vehicle      | `real_vehicle`             | Real      |
| Expanding    | `synth_blobs_expanding`    | Synthetic |
| Gradual      | `synth_blobs_gradual`      | Synthetic |
| Merge-Split  | `synth_blobs_merge_split`  | Synthetic |
| Sudden       | `synth_blobs_sudden`       | Synthetic |
| Fast-RBF     | `synth_rbf_fast`           | Synthetic |

#### Available Algorithms

| Canonical name    | Group       | Description                                   |
| ----------------- | ----------- | --------------------------------------------- |
| **AutoDSC**       | AutoML      | Our framework (PGSSO + composite CVI, no drift detector) |
| **ConfStream**    | AutoML      | Configuration-space stream AutoML             |
| **OACE**          | AutoML      | OnlineAutoClust Ensemble                      |
| **MOStream**      | Rule-based  | Multi-objective stream optimizer              |
| **RandomSearch**  | Baseline    | Random hyperparameter search                  |
| **Default(Best)** | Baseline    | Best fixed-default algorithm per dataset      |

Single fixed algorithms used internally: **CluStream, DenStream, DBSTREAM, StreamKMeans**.

### Running Individual Experiments

```bash
# Run AutoDSC on Insects dataset
python experiments/scripts/run_autodsc.py --dataset real_insects

# Run CluStream on PenDigits dataset
python experiments/scripts/run_clustream.py --dataset real_pendigits

# Run with custom parameters
python experiments/scripts/run_autodsc.py --dataset real_insects --window_size 500 --population_size 15
```

### Sensitivity Analysis

```bash
# Run sensitivity analysis for a specific parameter
python experiments/run_sensitivity_analysis.py --mode parameter --param gamma --dataset real_insects

# Run multi-parameter sensitivity analysis
python experiments/run_sensitivity_analysis.py --mode dataset --dataset real_insects

# Run full sensitivity analysis
python experiments/run_sensitivity_analysis.py --mode full --dataset real_insects

# Run random parameter search
python experiments/run_sensitivity_analysis.py --mode random --dataset real_insects --configs 10
```

### Ablation Study

Leave-one-out ablation of AutoDSC's key components. The variants are implemented
as subclasses in `experiments/scripts/ablation/AutoDSC_ablation.py`, so
`src/autodsc/AutoDSC.py` is never modified and every variant shares the same code
path except for the single component under study.

```bash
# Run a single ablation variant on a dataset
python experiments/run_autodsc_ablation.py --dataset real_insects --ablation no_recency

# Sanity check: `full` must reproduce plain AutoDSC
python experiments/run_autodsc_ablation.py --dataset real_insects --ablation full

# With custom AutoDSC settings
python experiments/run_autodsc_ablation.py --dataset real_insects --ablation single_cvi \
    --window_size 500 --population_size 15
```

**Available ablation variants:**

| Variant         | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| `full`          | Complete AutoDSC (sanity check, identical to AutoDSC)            |
| `no_recency`    | Raw-score elite ranking (drop recency weighting of the archive)  |
| `no_adaptive`   | Fixed exploration rate (drop adaptive explore/exploit)           |
| `single_cvi`    | Silhouette-only scoring (drop the three-CVI composite)           |
| `no_diversity`  | Score-only top-N pruning (drop diversity-preserving pruning)     |

Pure random search (no elite guidance) is intentionally not an ablation variant —
it is already covered by the `RandomSearchStream` baseline.

## Metrics

- **ARI** (Adjusted Rand Index) — primary quality metric, chance-adjusted, ↑ higher is better
- **CMM** (Cluster Mapping Measure) — stream-aware quality metric, ↑ higher is better
- **Throughput** (instances/s, ↑), **Latency** (µs/point, ↓), **Memory** (MB, ↓) — efficiency metrics

## Rust Implementation

For maximum performance, compile and use the Rust implementation:

```bash
# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Compile
cd src/stream_clustering
pip install maturin
maturin develop --release
cd ../..

# Verify
python -c "import stream_clustering; print('Rust implementation loaded!')"
```

## Output and Results

### What is tracked in this repository

Only code and the generated figures are version-controlled. **No JSON is tracked** —
every result file is reproducible from the runners above, so the repository stays
small and the numbers stay verifiable by re-running rather than by trusting a blob.

| Path | Contents | Tracked |
| ---- | -------- | ------- |
| `results/analysis/plot_*.py`, `style.py` | Chart scripts and the canonical colour/marker/order definitions | Yes |
| `results/analysis/charts/` | Generated figures (vector PDF + 300-dpi PNG) | Yes |
| `results/benchmark/`, `results/ablation/`, `results/sensitivity/` | Raw per-run metrics, one JSON per method × dataset × seed | No |
| `results/analysis/*.json`, `results/analysis/extra/` | Intermediate analysis data (alpha traces, evolution capture, dimensionality sweep, overhead, validity) | No |

### Regenerating the results

```bash
# 1. Raw benchmark / ablation / sensitivity runs -> results/{benchmark,ablation,sensitivity}/
python experiments/run_scripts.py --mode all
python experiments/run_autodsc_ablation.py --dataset real_insects --ablation no_recency
python experiments/run_sensitivity_analysis.py --mode full --dataset real_insects

# 2. Intermediate analysis data -> results/analysis/{,extra/}*.json
python experiments/run_alpha_trace.py
python results/analysis/run_extra_analysis.py

# 3. Figures -> results/analysis/charts/
python results/analysis/plot_benchmark_stages.py
python results/analysis/plot_cd.py
# ... and the remaining plot_*.py scripts
```

Step 1 is the expensive one — the full grid is 10 datasets x 10 seeds per method.
Steps 2 and 3 are cheap once the raw JSON exists.

### Chart directories

| Directory | Contents |
| --------- | -------- |
| `benchmark/`   | Main ARI/CMM comparison charts |
| `cd/`          | Critical difference diagrams |
| `ablation/`    | Ablation study visualisations |
| `sensitivity/` | Sensitivity analysis plots |
| `efficiency/`  | Throughput / latency / memory charts |
| `overhead/`    | AutoML search overhead |
| `evolution/`   | Population evolution over time |
| `recovery/`    | Drift recovery charts |
| `alpha/`       | Adaptive exploration-rate traces |
| `dimscaling/`  | Scaling with dimensionality |
| `validity/`    | Internal vs. external validity agreement |

## Citation

This work has been accepted at the **IEEE International Conference on Data Mining
(ICDM) 2026**. Please cite it as:

```bibtex
@inproceedings{verma2026autodsc,
  title     = {{AutoDSC}: Automated Algorithm Selection and Tuning for
               Non-Stationary Data Stream Clustering},
  author    = {Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard and Bahri, Maroua},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Data Mining (ICDM)},
  publisher = {IEEE},
  address   = {Shenyang, China},
  year      = {2026},
  url       = {https://github.com/TechyNilesh/AutoDSC}
}
```
