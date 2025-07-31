# AutoDSC: Automated Data Stream Clustering Framework

AutoDSC is a fast and scalable AutoML framework for real-time stream clustering that automatically selects and optimizes clustering algorithms for data streams with concept drift detection.

## Project Structure

```
AutoDSC/
├── AutoDSC.py                  # Main AutoDSC framework implementation
├── ConfStream.py              # ConfStream clustering algorithm
├── DAWIDD.py                  # Drift detection algorithm
├── OnlineAutoClust.py         # Online AutoClust implementation
├── RandomSearchStream.py      # Random search baseline
├── run_scripts.py             # Main experiment runner
├── run_sensitivity_analysis.py # Sensitivity analysis experiments
├── requirements.txt           # Python dependencies
├── scripts/                   # Individual algorithm runners
│   ├── run_autodsc.py
│   ├── run_clustream.py
│   ├── run_confstream.py
│   ├── run_dbstream.py
│   ├── run_denstream.py
│   ├── run_kmeans.py
│   ├── run_onlineautoclust.py
│   ├── run_randomsearch.py
│   ├── run_streamkmeans.py
│   └── utils/
│       └── purityscore.py
├── stream_clustering/         # Fast Rust implementation
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
├── stream_datasets/           # Benchmark datasets
│   ├── adult.csv
│   ├── electricity.csv
│   ├── forest_cover.csv
│   ├── insects.csv
│   └── ...
├── CHARTS/                    # Generated visualization charts
├── ABILATION_CHARTS/          # Ablation study results
└── SENSITIVITY_CHARTS/        # Sensitivity analysis results
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
# Note: Some algorithms may have slightly different names in River
```

#### Rust Implementation Version (Fast):
```python
from stream_clustering import CluStream, DenStream, DBStream, KMeans, StreamKMeans
```

The Rust version requires compilation but offers significantly better performance, especially on large datasets.

### Basic Usage

#### Using the Main Runner Script

The `run_scripts.py` provides multiple execution modes:

```bash
# Run all models on a specific dataset
python run_scripts.py --mode dataset --name adult

# Run a specific model on all datasets
python run_scripts.py --mode model --name clustream

# Run all models on all datasets
python run_scripts.py --mode all

# Run comprehensive evaluation with multiple runs
python run_scripts.py --mode random
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
| `synthetic_blobs_100k_samples_5features_8clusters` | 100,000 | 5 | 8 | Synthetic blobs dataset |
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

You can also run individual algorithm scripts:

```bash
# Run AutoDSC on insects dataset
python scripts/run_autodsc.py --dataset insects

# Run CluStream on adult dataset
python scripts/run_clustream.py --dataset adult
```

### Advanced Usage

#### Custom Parameters

```bash
# Custom progress interval and output directory
python run_scripts.py --mode dataset --name adult --progress_interval 500 --output_dir my_results
```

#### Sensitivity Analysis

```bash
# Run sensitivity analysis for AutoDSC parameters
python run_sensitivity_analysis.py
```

## Rust Implementation Performance

For maximum performance, especially on large datasets, compile and use the Rust implementation:

1. **Prerequisites:**
   - Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   - Install maturin: `pip install maturin`

2. **Compilation:**
   ```bash
   cd stream_clustering
   maturin develop --release
   ```

3. **Verification:**
   ```python
   import stream_clustering
   print("Rust implementation loaded successfully!")
   ```

## Output and Results

- Results are saved in the `benchmark_results/` directory (or custom directory specified)
- Performance metrics include Rand Index, runtime, and memory usage
- Charts and visualizations are generated in `CHARTS/`, `ABILATION_CHARTS/`, and `SENSITIVITY_CHARTS/`

## Key Features

- **Automated Algorithm Selection:** Automatically selects the best clustering algorithm for your data stream
- **Concept Drift Detection:** Built-in DAWIDD drift detector for handling evolving data streams
- **Population-Guided Search:** PGSSO optimizer for efficient hyperparameter optimization
- **Fast Rust Implementation:** Optional high-performance Rust backend for large-scale experiments
- **Comprehensive Evaluation:** Built-in evaluation metrics and visualization tools
- **Multiple Baseline Algorithms:** Includes implementations of state-of-the-art stream clustering algorithms
