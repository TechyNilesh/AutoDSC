# AutoDSC: An Automated Data Stream Clustering Framework

This repository contains the implementation of **AutoDSC** (**Auto**mated **D**ata **S**tream **C**lustering), a novel AutoML framework that automatically optimizes algorithm selection and hyperparameter tuning for real-time stream clustering.

Data stream clustering faces unique challenges in dynamic environments where data distributions evolve continuously and computational resources are limited. AutoDSC addresses these challenges by employing **Population-Guided Stochastic Search Optimization (PGSSO)**, leveraging elite configurations to guide efficient parameter exploration while dynamically adjusting exploration rates based on performance feedback. The framework incorporates **drift-aware adaptation** using DAWIDD for proactive response to distributional changes and **multi-objective evaluation** with clustering validation indices for robust performance assessment.

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
│   ├── adult.csv
│   ├── electricity.csv
│   ├── forest_cover.csv
│   ├── insects.csv
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
│   ├── charts/
│   │   ├── ablation/             # Ablation study charts
│   │   ├── comparison/           # Baseline comparison charts
│   │   └── sensitivity/          # Sensitivity analysis charts
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

The project provides two identical implementations with the same functionality - the only difference is the import source:

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
python experiments/run_scripts.py --mode dataset --name adult

# Run a specific model on all datasets
python experiments/run_scripts.py --mode model --name clustream

# Run all models on all datasets
python experiments/run_scripts.py --mode all

# Run comprehensive evaluation with multiple runs
python experiments/run_scripts.py --mode random
```

#### Available Datasets

| Dataset Name | Instances | Attributes | Clusters | Description |
|-------------|-----------|------------|----------|-------------|
| `adult` | 48,842 | 14 | 4 | Adult Census dataset |
| `electricity` | 45,312 | 8 | 2 | Electricity market dataset |
| `forest_cover` | 581,012 | 54 | 7 | Forest cover type dataset |
| `hyperplane_high_gradual_drift` | 500,000 | 10 | 2 | Synthetic gradual drift |
| `insects` | 52,848 | 33 | 6 | Insects dataset |
| `movingRBF` | 200,000 | 10 | 5 | Moving RBF synthetic dataset |
| `moving_squares` | 200,000 | 2 | 4 | Moving squares synthetic dataset |
| `new_airlines` | 539,383 | 7 | 2 | Airlines dataset |
| `sea_high_abrupt_drift` | 500,000 | 3 | 2 | SEA with abrupt drift |
| `sea_high_mixed_drift` | 500,000 | 3 | 2 | SEA with mixed drift |
| `synth_RandomRBFDrift` | 100,000 | 4 | 4 | Random RBF drift |
| `vehicle_sensIT` | 98,528 | 100 | 3 | Vehicle sensor dataset |

#### Available Algorithms
- `autodsc` - AutoDSC (our framework)
- `clustream` - CluStream
- `dbstream` - DBSTREAM
- `denstream` - DenStream
- `kmeans` - K-Means
- `streamkmeans` - StreamKMeans
- `confstream` - ConfStream
- `onlineautoclust` - OnlineAutoClust
- `randomsearch` - Random Search baseline

### Running Individual Experiments

```bash
# Run AutoDSC on insects dataset
python experiments/scripts/run_autodsc.py --dataset insects

# Run CluStream on adult dataset
python experiments/scripts/run_clustream.py --dataset adult

# Run with custom parameters
python experiments/scripts/run_autodsc.py --dataset insects --window_size 500 --population_size 15
```

### Sensitivity Analysis

```bash
# Run sensitivity analysis for a specific parameter
python experiments/run_sensitivity_analysis.py --mode parameter --param gamma --dataset insects

# Run multi-parameter sensitivity analysis
python experiments/run_sensitivity_analysis.py --mode dataset --dataset insects

# Run full sensitivity analysis
python experiments/run_sensitivity_analysis.py --mode full --dataset insects

# Run random parameter search
python experiments/run_sensitivity_analysis.py --mode random --dataset insects --configs 10
```

### Ablation Study

```bash
# Run ablation study for a specific component
python experiments/run_ablation_study.py --dataset insects --ablation drift_detection

# Run all ablations
python experiments/run_ablation_study.py --dataset insects --ablation all

# Run full comparison (baseline + all ablations)
python experiments/run_ablation_study.py --dataset insects --ablation full_comparison

# Run on all datasets
python experiments/run_ablation_study.py --ablation drift_detection --all_datasets
```

**Available ablation options:**
| Ablation | Description |
|----------|-------------|
| `drift_detection` | Disable DAWIDD drift detection |
| `elite_guidance` | Disable PGSSO elite guidance (random sampling only) |
| `adaptive_exploration_rate` | Disable adaptive exploration rate (fixed rate) |
| `diversity_maintenance` | Disable diversity maintenance in population |
| `all` | Run all ablations |
| `full_comparison` | Run baseline + all ablations |

## AutoDSC Components

### Core Algorithm (`algorithms/AutoDSC/`)

1. **AutoDSC.py** - Main framework with:
   - Population-Guided Stochastic Search Optimizer (PGSSO)
   - Adaptive exploration rate
   - Multi-objective scoring (Silhouette, Calinski-Harabasz, Davies-Bouldin)
   - Diversity maintenance

2. **DAWIDD.py** - Dynamic Adapting Window Independence Drift Detector:
   - Detects concept drift in data streams
   - Uses statistical independence testing
   - Triggers model reset on drift detection

3. **svm_test.py** - SVM-based independence test:
   - Tests if data is time-dependent
   - Used by DAWIDD for drift detection

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

- Benchmark results: `benchmark_results/` (or custom directory)
- Charts: `results/charts/`
  - `ablation/` - Ablation study visualizations
  - `comparison/` - Baseline comparison charts
  - `sensitivity/` - Sensitivity analysis plots
- Tables: `results/tables/` - CSV result files

## Key Features

- **Automated Algorithm Selection** - Automatically selects the best clustering algorithm
- **Concept Drift Detection** - Built-in DAWIDD detector for evolving data streams
- **Population-Guided Search** - PGSSO optimizer for hyperparameter optimization
- **Adaptive Exploration** - Dynamic exploration/exploitation balance
- **Fast Rust Backend** - High-performance implementation for large-scale experiments
- **Comprehensive Evaluation** - Built-in metrics and visualization tools

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{anonymous2026autodsc,
  title={AutoDSC: An Automated Data Stream Clustering Framework},
  author={Anonymous},
  booktitle={Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)},
  year={2026},
  note={Under review}
}
```

## License

This project is released for academic and research purposes. Please refer to the LICENSE file for details.
