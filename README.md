# AutoDSC: Automated Algorithm Selection and Tuning for Non-Stationary Data Stream Clustering

This repository contains the implementation of **AutoDSC** (**Auto**mated **D**ata **S**tream **C**lustering), an AutoML framework that continuously searches the joint algorithm-and-hyperparameter space for real-time stream clustering.

Data stream clustering must adapt to evolving distributions without ground-truth labels, so no single algorithm or hyperparameter configuration remains effective throughout a stream. AutoDSC addresses this by employing **Population-Guided Stochastic Search Optimisation (PGSSO)**, which combines a recency-weighted elite archive with adaptive exploration. A composite of three internal cluster-validity indices enables label-free configuration selection, while **DAWIDD** drift detection allows proactive response to distributional changes. Across five real-world and five synthetic streams, AutoDSC attains the best mean rank on both Adjusted Rand Index (ARI) and Cluster Mapping Measure (CMM) while remaining computationally competitive with stream-clustering AutoML baselines.

## Project Structure

```
AutoDSC/
├── algorithms/                    # Clustering algorithm implementations
│   ├── AutoDSC/
│   │   ├── AutoDSC.py            # Main AutoDSC framework
│   │   ├── DAWIDD.py             # Drift detection algorithm
│   │   └── svm_test.py           # SVM-based independence test for drift detection
│   ├── ConfStream/
│   │   └── ConfStream.py
│   ├── OnlineAutoClust/
│   │   └── OnlineAutoClust.py
│   └── RandomSearchStream/
│       └── RandomSearchStream.py
├── datasets/                      # Benchmark datasets
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
│   └── ...
├── experiments/                   # Experiment runners
│   ├── run_ablation_study.py     # Ablation study experiments
│   ├── run_scripts.py            # Main experiment runner
│   ├── run_sensitivity_analysis.py
│   └── scripts/                  # Individual algorithm runners
│       ├── run_autodsc.py
│       ├── run_clustream.py
│       ├── run_confstream.py
│       ├── run_dbstream.py
│       ├── run_denstream.py
│       ├── run_kmeans.py
│       ├── run_onlineautoclust.py
│       ├── run_randomsearch.py
│       ├── run_streamkmeans.py
│       └── utils/
│           └── purityscore.py
├── results/                       # Experiment results
│   ├── analysis/
│   │   ├── charts/               # Generated PDF/PNG figures
│   │   │   ├── ablation/
│   │   │   ├── benchmark/
│   │   │   ├── cd/
│   │   │   ├── dimscaling/
│   │   │   ├── efficiency/
│   │   │   ├── evolution/
│   │   │   ├── overhead/
│   │   │   ├── recovery/
│   │   │   ├── sensitivity/
│   │   │   └── validity/
│   │   └── style.py              # Canonical colors, markers, method order
│   └── tables/                   # Result tables (CSV)
├── stream_clustering/             # Fast Rust implementation
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       ├── lib.rs
│       ├── clustream.rs
│       ├── dbstream.rs
│       ├── denstream.rs
│       ├── kmeans.rs
│       ├── streamkmeans.rs
│       └── ...
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
   git clone <repository-url>
   cd AutoDSC
   ```
2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Compile Rust implementation (for faster performance):**

   ```bash
   cd stream_clustering
   pip install maturin
   maturin develop --release
   cd ..
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
| **AutoDSC**       | AutoML      | Our framework (PGSSO + DAWIDD + validity index) |
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

```bash
# Run ablation study for a specific component
python experiments/run_ablation_study.py --dataset real_insects --ablation drift_detection

# Run all ablations
python experiments/run_ablation_study.py --dataset real_insects --ablation all

# Run full comparison (baseline + all ablations)
python experiments/run_ablation_study.py --dataset real_insects --ablation full_comparison

# Run on all datasets
python experiments/run_ablation_study.py --ablation drift_detection --all_datasets
```

**Available ablation options:**

| Ablation                      | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| `drift_detection`             | Disable DAWIDD drift detection                      |
| `elite_guidance`              | Disable PGSSO elite guidance (random sampling only) |
| `adaptive_exploration_rate`   | Disable adaptive exploration rate (fixed rate)      |
| `diversity_maintenance`       | Disable diversity maintenance in population         |
| `all`                         | Run all ablations                                   |
| `full_comparison`             | Run baseline + all ablations                        |

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
cd stream_clustering
pip install maturin
maturin develop --release
cd ..

# Verify
python -c "import stream_clustering; print('Rust implementation loaded!')"
```

## Output and Results

- Benchmark results: `results/analysis/`
- Charts: `results/analysis/charts/` (vector PDF + 300-dpi PNG)
  - `benchmark/` — main ARI/CMM comparison charts
  - `ablation/` — ablation study visualizations
  - `sensitivity/` — sensitivity analysis plots
  - `cd/` — critical difference diagrams
  - `efficiency/` — throughput/latency/memory charts
  - `evolution/` — population evolution over time
  - `recovery/` — drift recovery charts
- Tables: `results/tables/` — CSV result files

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{verma2026autodsc,
  title={AutoDSC: Automated Algorithm Selection and Tuning for Non-Stationary Data Stream Clustering},
  author={Anonymous},
  year={2026},
  note={Under review}
}
```
