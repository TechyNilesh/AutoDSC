#!/usr/bin/env python3

"""
AutoDSC: Automated Data Stream Clustering Framework

Authors: Anonymous
Date: June 2024
"""

import numpy as np
import random
import copy
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from DAWIDD import DAWIDD

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import stream_clustering module
try:
    import stream_clustering
except ImportError:
    print("Warning: stream_clustering module not found. Using fallback implementations.")

class DriftDetector:
    def __init__(self, max_window_size=25, min_window_size=10, min_p_value=0.001):
        """
        Initialize DAWIDD drift detector.
        
        Parameters:
        - max_window_size: Maximum samples in DAWIDD window
        - min_window_size: Minimum samples needed for testing
        - min_p_value: P-value threshold for independence test
        """
        self.detector = DAWIDD(
            max_window_size=max_window_size,
            min_window_size=min_window_size,
            min_p_value=min_p_value
        )
        self.drift_detected = False
        self.data_point_counter = 0  # Track time steps

    def detect_drift(self, current_window):
        """
        Update DAWIDD with window data, adding timestamps.
        Returns True if drift is detected in any data point.
        """
        self.drift_detected = False

        norms = np.linalg.norm(current_window, axis=1)
        mean = np.mean(norms)
        std = np.std(norms)
        min = np.min(norms)
        max = np.max(norms)
        median = np.median(norms)

        # Convert scalars to 1D arrays before concatenation
        features = np.array([mean, std, min, max, median, self.data_point_counter])
        data_with_time = features.reshape(1, -1)

        #print("Data With Time:",data_with_time)
        
        # Check for drift on this data point
        self.detector.add_batch(data_with_time)
        
        if self.detector.detected_change():
            self.drift_detected = True
            # Don't break here - continue processing remaining points
        
        self.data_point_counter += 1
        
        return self.drift_detected

# =============================================================================
# NORMALIZED OBJECTIVE FUNCTIONS FOR CLUSTERING EVALUATION
# =============================================================================

def normalized_silhouette_coefficient(data: np.ndarray, labels: List[int]) -> float:
    """
    Calculate normalized Silhouette Coefficient.
    Args:
        data: Input data points
        labels: Cluster labels for each data point
    Returns:
        Normalized silhouette score [0, 1] where higher is better
    """
    silhouette = silhouette_score(data, labels)
    normalized = (1 + silhouette) / 2  # Map [-1, 1] to [0, 1]
    return normalized

def normalized_davies_bouldin_index(data: np.ndarray, labels: List[int]) -> float:
    """
    Calculate normalized Davies-Bouldin Index.
    Paper: normalized as 1 - tanh(x), where x is the DB index.
    Args:
        data: Input data points
        labels: Cluster labels for each data point
    Returns:
        Normalized DB score [0, 1] where higher is better
    """
    davies_bouldin = davies_bouldin_score(data, labels)
    normalized = 1.0 - np.tanh(davies_bouldin)
    return normalized

def normalized_calinski_harabasz_index(data: np.ndarray, labels: List[int]) -> float:
    """
    Calculate normalized Calinski-Harabasz Index.
    Paper: normalized as tanh(x), where x is the CH index.
    Args:
        data: Input data points
        labels: Cluster labels for each data point
    Returns:
        Normalized CH score [0, 1] where higher is better
    """
    calinski_harabasz = calinski_harabasz_score(data, labels)
    normalized = np.tanh(np.log1p(calinski_harabasz) / 10.0)
    return normalized

def calculate_normalized_objective_scores(data: np.ndarray, labels: List[int]) -> Dict[str, float]:
    """
    Calculate all three normalized objective scores.
    Args:
        data: Input data points
        labels: Cluster labels for each data point
    Returns:
        Dictionary containing all normalized scores
    """
    return {
        'normalized_silhouette': normalized_silhouette_coefficient(data, labels),
        'normalized_davies_bouldin': normalized_davies_bouldin_index(data, labels),
        'normalized_calinski_harabasz': normalized_calinski_harabasz_index(data, labels)
    }

# =============================================================================
# DATA CLASSES FOR CONFIGURATION AND SCORING
# =============================================================================

@dataclass
class ParameterConfig:
    """Configuration for algorithm parameters."""
    name: str
    param_type: str  # 'numeric', 'integer', 'categorical', 'boolean'
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    categories: Optional[List] = None
    default: Any = None

@dataclass
class ObjectiveScore:
    """Base class for multi-objective scoring focusing on quality."""
    silhouette: float = 0.0  # Normalized silhouette: higher is better [0,1]
    calinski: float = 0.0    # Normalized Calinski-Harabasz: higher is better [0,1]
    davies: float = 0.0      # Normalized Davies-Bouldin: higher normalized is better [0,1]

    def composite_score(self) -> float:
        """
        Calculate composite score based on clustering quality metrics only.
        Returns:
            Mean of the three normalized clustering validation indices
        """
        return (self.silhouette + self.calinski + self.davies) / 3.0

@dataclass
class AlgorithmConfig:
    """Configuration for clustering algorithms."""
    algorithm_name: str
    algorithm_class: Any
    parameters: List[ParameterConfig]
    current_params: Dict = field(default_factory=dict)
    objective_scores: ObjectiveScore = field(default_factory=ObjectiveScore)
    model_instance: Any = None
    
    # New fields for continuous evaluation - using thread-safe collections
    window_labels: List = field(default_factory=list)

@dataclass
class EliteConfig:
    """Elite configuration for population-guided sampling."""
    config_dict: Dict
    score: ObjectiveScore
    frequency: int = 1
    last_seen: int = 0

# =============================================================================
# POPULATION-GUIDED STOCHASTIC SEARCH OPTIMIZER
# =============================================================================

class PGSSO:
    """Population-Guided Stochastic Search Optimizer for AutoDSC."""

    def __init__(self, elite_size: int = 5, exploration_rate: float = 0.5):
        """Initialize the optimizer."""
        self.elite_size = max(1, elite_size)  # Ensure minimum elite size
        self.elite_population = []
        self.exploration_rate = max(0.1, min(1.0, exploration_rate))  # Range: 0.1-1.0
        self.exploration_rate = self.exploration_rate
        self.generation_count = 0
        self.parameter_distributions = {}

    def update_elite_population(self, configs: List[Dict], scores: List[ObjectiveScore]):
        """Update elite population with new configurations and their scores."""
        if not configs or not scores or len(configs) != len(scores):
            return

        # Create new elite configurations
        new_elites = []
        for config, score in zip(configs, scores):
            # Validate score
            if score is None or not hasattr(score, 'composite_score'):
                continue
                
            composite = score.composite_score()
            if not np.isfinite(composite):
                continue

            elite = EliteConfig(
                config_dict=copy.deepcopy(config),
                score=score,
                last_seen=self.generation_count
            )
            new_elites.append(elite)

        # Merge with existing elites and keep top performers
        all_elites = self.elite_population + new_elites
        
        # Sort by composite score, handling any potential errors
        valid_elites = []
        for elite in all_elites:
            try:
                score = elite.score.composite_score()
                if np.isfinite(score):
                    valid_elites.append(elite)
            except:
                continue

        valid_elites.sort(key=lambda x: x.score.composite_score(), reverse=True)
        self.elite_population = valid_elites[:self.elite_size]

        # Update parameter distributions for guided sampling
        self._update_parameter_distributions()
        self.generation_count += 1

    def _update_parameter_distributions(self):
        """Update parameter distributions based on current elite population."""
        if not self.elite_population:
            return

        self.parameter_distributions = {}
        algorithm_param_groups = defaultdict(lambda: defaultdict(list))

        # Group parameters by algorithm and name
        for elite in self.elite_population:
            if not elite.config_dict:
                continue
            algorithm_name = elite.config_dict.get('algorithm')
            if not algorithm_name:
                continue
            for param_name, param_value in elite.config_dict.items():
                if param_name != 'algorithm' and param_value is not None:
                    algorithm_param_groups[algorithm_name][param_name].append(param_value)

        # Calculate distributions for each algorithm's parameters
        for algorithm_name, param_groups in algorithm_param_groups.items():
            if algorithm_name not in self.parameter_distributions:
                self.parameter_distributions[algorithm_name] = {}
                
            for param_name, values in param_groups.items():
                if not values:
                    continue
                    
                if all(isinstance(v, (int, float)) for v in values):
                    # Numerical parameter: use Gaussian distribution
                    mean_val = np.mean(values)
                    std_val = max(0.01, np.std(values))
                    if np.isfinite(mean_val) and np.isfinite(std_val):
                        self.parameter_distributions[algorithm_name][param_name] = {
                            'type': 'numerical',
                            'mean': mean_val,
                            'std': std_val,
                            'values': values
                        }
                else:
                    # Categorical parameter: use probability distribution
                    unique_values, counts = np.unique(values, return_counts=True)
                    probabilities = counts / len(values)
                    self.parameter_distributions[algorithm_name][param_name] = {
                        'type': 'categorical',
                        'values': unique_values.tolist(),
                        'probabilities': probabilities.tolist()
                    }
           

    def generate_config(self, algorithm_configs: List[AlgorithmConfig]) -> Optional[Dict]:
        """Generate new configuration using population-guided sampling."""
        if not algorithm_configs:
            return None

        # Decide between exploration (random) and exploitation (elite-guided)
        if not self.elite_population or random.random() < self.exploration_rate:
            return self._random_config(algorithm_configs)
        else:
            return self._sample_from_elite(algorithm_configs)

    def _sample_from_elite(self, algorithm_configs: List[AlgorithmConfig]) -> Dict:
        """Sample configuration from elite population distributions."""
        # Choose algorithm based on elite population preferences
        algorithm_counts = defaultdict(int)
        available_algorithms = {config.algorithm_name for config in algorithm_configs}
        
        # Count elite algorithms that are available
        for elite in self.elite_population:
            algorithm_name = elite.config_dict.get('algorithm')
            if algorithm_name in available_algorithms:
                algorithm_counts[algorithm_name] += 1

        # If no elite algorithms are available, weight equally but still use elite parameters
        if not algorithm_counts:
            for config in algorithm_configs:
                algorithm_counts[config.algorithm_name] = 1

        algorithms = list(algorithm_counts.keys())
        weights = [algorithm_counts[alg] for alg in algorithms]
        
        # Select algorithm probabilistically based on elite performance
        chosen_algorithm = np.random.choice(algorithms, p=np.array(weights)/sum(weights))

        # Find the corresponding algorithm configuration
        algorithm_config = None
        for alg_config in algorithm_configs:
            if alg_config.algorithm_name == chosen_algorithm:
                algorithm_config = alg_config
                break

        if not algorithm_config:
            return self._random_config(algorithm_configs)

        # Generate parameters based on elite distributions
        config_dict = {'algorithm': chosen_algorithm}
        for param in algorithm_config.parameters:
            param_name = param.name
            if (chosen_algorithm in self.parameter_distributions and 
                param_name in self.parameter_distributions[chosen_algorithm]):
                dist = self.parameter_distributions[chosen_algorithm][param_name]
                if dist['type'] == 'numerical':
                    # Sample from Gaussian distribution around elite mean
                    sampled_value = np.random.normal(dist['mean'], dist['std'])
                    
                    # Apply parameter constraints
                    if param.range_min is not None and param.range_max is not None:
                        sampled_value = np.clip(sampled_value, param.range_min, param.range_max)
                    
                    # Always check for integer type, regardless of range constraints
                    if param.param_type == 'integer':
                        sampled_value = int(round(sampled_value))
                    config_dict[param_name] = sampled_value

                elif dist['type'] == 'categorical':
                    # Sample from categorical distribution
                    config_dict[param_name] = np.random.choice(
                        dist['values'], p=dist['probabilities']
                    )
                else:
                    config_dict[param_name] = param.default

            else:
                # No elite guidance available, use random sampling
                config_dict[param_name] = self._sample_parameter_randomly(param)

        return config_dict

    def _random_config(self, algorithm_configs: List[AlgorithmConfig]) -> Dict:
        """Generate completely random configuration for exploration."""
        config = random.choice(algorithm_configs)
        params = {'algorithm': config.algorithm_name}
        for param in config.parameters:
            params[param.name] = self._sample_parameter_randomly(param)
        return params

    def _sample_parameter_randomly(self, param: ParameterConfig) -> Any:
        """Sample a parameter value randomly based on its type and constraints."""
        if param.param_type == 'integer' and param.range_min is not None and param.range_max is not None:
            return random.randint(int(param.range_min), int(param.range_max))
        elif param.param_type == 'numeric' and param.range_min is not None and param.range_max is not None:
            return random.uniform(param.range_min, param.range_max)
        elif param.param_type == 'categorical' and param.categories:
            return random.choice(param.categories)
        elif param.param_type == 'boolean':
            return random.choice([True, False])
        else:
            return param.default

    def get_exploration_stats(self) -> Dict:
        """Get statistics about current exploration behavior."""
        return {
            'exploration_rate': self.exploration_rate,
            'elite_population_size': len(self.elite_population),
            'generation_count': self.generation_count,
            'best_score': self.elite_population[0].score.composite_score() if self.elite_population else 0.0
        }

# =============================================================================
# MAIN AUTOSTREAMCLUST CLASS
# =============================================================================

class AutoDSC:
    """
    AutoDSC: A Fast and Scalable AutoML Framework for Real-Time Stream Clustering

    Redesigned search: Tournament-based selection on window data.
    - First window: run tournament to find best config
    - Between windows: only train the winning model (fast)
    - Each window: check quality, re-run tournament if drift detected
    """

    def __init__(self,
                 window_size: int = 1000,
                 population_size: int = 20,
                 random_seed: int = 42,
                 gamma: float = 0.1,
                 diversity_threshold: float = 0.1,
                 **kwargs):
        self.window_size = window_size
        self.population_size = kwargs.get('population_size', max(2, population_size))
        self.elite_size = max(1, self.population_size // 2)
        self.gamma = kwargs.get('gamma', max(0.01, min(1.0, gamma)))
        self.diversity_threshold = kwargs.get('diversity_threshold', max(0.01, min(1.0, diversity_threshold)))
        self.sample_ratio = 0.25

        random.seed(kwargs.get('random_seed', random_seed))
        np.random.seed(kwargs.get('random_seed', random_seed))

        self.optimizer = PGSSO(
            elite_size=self.elite_size,
            exploration_rate=0.5
        )

        self.algorithm_configs = self._initialize_algorithm_configs()

        self.configuration_archive = []
        self.current_window = []
        self.window_count = 0
        self.incumbent = None
        self._previous_score = -float('inf')
        self._drift_threshold = 0.25  # Quality drop threshold for re-search (conservative)
        self._replay_buffer = []  # Fix 6: warm-start new configs with recent window history
        self._prev_window_predictions = {}  # Fix B: {config_id: predictions} for stability scoring

        self.drift_detector = DriftDetector(
            max_window_size=kwargs.get('max_window_size', 25),
            min_window_size=kwargs.get('min_window_size', 10),
            min_p_value=kwargs.get('min_p_value', 0.001),
        )

        self._initialize_archive()

    def __str__(self):
        """String representation of the class."""
        return "AutoDSC"

    def _validate_config(self, config_dict: Dict) -> bool:
        """Validate configuration dictionary before creating instances."""
        if not isinstance(config_dict, dict):
            return False

        required_keys = ['algorithm']
        if not all(key in config_dict for key in required_keys):
            return False

        algorithm_name = config_dict['algorithm']
        if algorithm_name not in [cfg.algorithm_name for cfg in self.algorithm_configs]:
            return False

        # Validate parameter values
        algorithm_config = None
        for cfg in self.algorithm_configs:
            if cfg.algorithm_name == algorithm_name:
                algorithm_config = cfg
                break

        if not algorithm_config:
            return False

        # Check parameter constraints
        for param in algorithm_config.parameters:
            if param.name in config_dict:
                value = config_dict[param.name]
                
                # Type checking with conversion for integers
                if param.param_type == 'integer':
                    if isinstance(value, float):
                        # Convert float to int if it's a whole number
                        if value.is_integer():
                            config_dict[param.name] = int(value)
                            value = config_dict[param.name]
                        else:
                            return False
                    elif not isinstance(value, int):
                        return False
                elif param.param_type == 'numeric' and not isinstance(value, (int, float)):
                    return False
                elif param.param_type == 'boolean':
                    if not isinstance(value, bool):
                        # Convert common boolean representations
                        if isinstance(value, (int, float)) and value in [0, 1]:
                            config_dict[param.name] = bool(value)
                            value = config_dict[param.name]
                        else:
                            return False
                elif param.param_type == 'categorical' and param.categories and value not in param.categories:
                    return False

                # Range checking
                if param.range_min is not None and value < param.range_min:
                    return False
                if param.range_max is not None and value > param.range_max:
                    return False

        return True

    def _initialize_algorithm_configs(self) -> List[AlgorithmConfig]:
        """Initialize clustering algorithm configurations with their parameter spaces."""
        configs = []

        algorithm_specs = {
            "CluStream": {
                "class": stream_clustering.CluStream,
                "params": [
                    ParameterConfig("n_macro_clusters", "integer", 2, 10, default=5),
                    ParameterConfig("max_micro_clusters", "integer", 5, 50, default=10),
                    ParameterConfig("time_window", "integer", 100, 2000, default=1000),
                    ParameterConfig("micro_cluster_r_factor", "numeric", 1.0, 5.0, default=2.0),
                    ParameterConfig("time_gap", "integer", 1, 100, default=10),
                ]
            },
            "DenStream": {
                "class": stream_clustering.DenStream,
                "params": [
                    ParameterConfig("decaying_factor", "numeric", 0.01, 0.5, default=0.1),
                    ParameterConfig("beta", "numeric", 0.5, 1.0, default=0.5),
                    ParameterConfig("mu", "numeric", 2.0, 10.0, default=4.0),
                    ParameterConfig("epsilon", "numeric", 0.01, 0.5, default=0.02),
                    ParameterConfig("n_samples_init", "integer", 10, 1000, default=100),
                    ParameterConfig("stream_speed", "integer", 10, 1000, default=100),
                ]
            },
            "DBSTREAM": {
                "class": stream_clustering.DBStream,
                "params": [
                    ParameterConfig("clustering_threshold", "numeric", 0.1, 2.0, default=1.0),
                    ParameterConfig("fading_factor", "numeric", 0.001, 0.1, default=0.01),
                    ParameterConfig("cleanup_interval", "integer", 1, 100, default=10),
                    ParameterConfig("intersection_factor", "numeric", 0.1, 0.9, default=0.3),
                    ParameterConfig("minimum_weight", "numeric", 0.5, 5.0, default=1.0),
                ]
            },
            "KMeans": {
                "class": stream_clustering.KMeans,
                "params": [
                    ParameterConfig("n_clusters", "integer", 2, 30, default=5),
                    ParameterConfig("halflife", "numeric", 0.01, 1.0, default=0.5),
                    ParameterConfig("mu", "numeric", 0.0, 1.0, default=0.0),
                    ParameterConfig("sigma", "numeric", 0.1, 2.0, default=1.0),
                    ParameterConfig("p", "categorical", categories=[1, 2], default=2),
                ]
            }
        }

        for name, spec in algorithm_specs.items():
            configs.append(AlgorithmConfig(name, spec["class"], spec["params"]))

        return configs

    def _initialize_archive(self):
        """Initialize configuration archive with random configurations."""
        for _ in range(self.population_size):
            config_dict = self.optimizer._random_config(self.algorithm_configs)
            if self._validate_config(config_dict):
                config = self._dict_to_config(config_dict)
                if config and config.model_instance is not None:
                    self.configuration_archive.append(config)

    def _create_algorithm_instance(self, config: AlgorithmConfig):
        """Create algorithm instance with given configuration."""
        param_mapping = {
            "CluStream": {
                "n_macro_clusters": config.current_params.get("n_macro_clusters", 5),
                "max_micro_clusters": config.current_params.get("max_micro_clusters", 50),
                "time_window": config.current_params.get("time_window", 1000),
                "micro_cluster_r_factor": config.current_params.get("micro_cluster_r_factor", 2.0),
                "time_gap": config.current_params.get("time_gap", 10)
            },
            "DenStream": {
                "decaying_factor": config.current_params.get("decaying_factor", 0.1),
                "beta": config.current_params.get("beta", 0.4),
                "mu": config.current_params.get("mu", 4.0),
                "epsilon": config.current_params.get("epsilon", 0.02),
                "n_samples_init": config.current_params.get("n_samples_init", 100),
                "stream_speed": config.current_params.get("stream_speed", 100)
            },
            "DBSTREAM": {
                "clustering_threshold": config.current_params.get("clustering_threshold", 1.0),
                "fading_factor": config.current_params.get("fading_factor", 0.01),
                "cleanup_interval": config.current_params.get("cleanup_interval", 10),
                "intersection_factor": config.current_params.get("intersection_factor", 0.3),
                "minimum_weight": config.current_params.get("minimum_weight", 1.0)
            },
            "KMeans": {
                "n_clusters": config.current_params.get("n_clusters", 5),
                "halflife": config.current_params.get("halflife", 0.5),
                "mu": config.current_params.get("mu", 0.0),
                "sigma": config.current_params.get("sigma", 1.0),
                "p": config.current_params.get("p", 2)
            },
        }

        if config.algorithm_name in param_mapping:
            params = param_mapping[config.algorithm_name]
            # Validate parameters before creating instance
            for key, value in params.items():
                if not np.isfinite(value) if isinstance(value, (int, float)) else False:
                    print(f"Warning: Invalid parameter {key}={value} for {config.algorithm_name}")
                    return None

            return config.algorithm_class(**params)
        else:
            return None

    def _dict_to_config(self, config_dict: Dict) -> Optional[AlgorithmConfig]:
        """Convert configuration dictionary to AlgorithmConfig object."""
        if not config_dict or 'algorithm' not in config_dict:
            return None

        algorithm_name = config_dict.get('algorithm')

        # Find base configuration for this algorithm
        base_config = None
        for alg_config in self.algorithm_configs:
            if alg_config.algorithm_name == algorithm_name:
                base_config = alg_config
                break

        if not base_config:
            return None

        # Create new configuration
        new_config = AlgorithmConfig(
            algorithm_name=base_config.algorithm_name,
            algorithm_class=base_config.algorithm_class,
            parameters=copy.deepcopy(base_config.parameters),
            current_params={k: v for k, v in config_dict.items() if k != 'algorithm'},
            objective_scores=ObjectiveScore(),
            model_instance=None
        )

        # Create algorithm instance
        new_config.model_instance = self._create_algorithm_instance(new_config)

        return new_config if new_config.model_instance else None

    def learn_one(self, x):
        """Process one instance - train all models incrementally."""
        if x is None:
            raise ValueError("Input cannot be None")

        x_array = np.asarray(x).flatten()
        if x_array.size == 0:
            raise ValueError("Input array is empty")
        if not np.all(np.isfinite(x_array)):
            return

        self.current_window.append(x_array.copy())

        # Train ALL models incrementally (they accumulate knowledge over time)
        for config in self.configuration_archive:
            if config.model_instance is None:
                config.model_instance = self._create_algorithm_instance(config)
                if config.model_instance is None:
                    continue
            try:
                config.model_instance.learn_one(x_array)
            except Exception:
                pass

        # Process window when full
        if len(self.current_window) == self.window_size:
            self._process_window()

    def predict_one(self, x):
        """Make prediction using the incumbent, or fallback to first available model."""
        active = self.incumbent
        if not active or active.model_instance is None:
            # Fallback: use first archive model that has been training
            for config in self.configuration_archive:
                if config.model_instance is not None:
                    active = config
                    break
            if not active or active.model_instance is None:
                return 0

        x_array = np.asarray(x).flatten()
        if x_array.size == 0 or not np.all(np.isfinite(x_array)):
            return 0

        try:
            return active.model_instance.predict_one(x_array)
        except Exception:
            return 0

    def _stability_score(self, config_id: str, current_preds: list, eval_data: np.ndarray) -> float:
        """Fix B: measure cross-window label consistency using ARI between consecutive windows.
        Configs that flip assignments every window (like DBSTREAM) score near 0.
        Returns a stability score in [0, 1].
        """
        from sklearn.metrics import adjusted_rand_score
        if config_id not in self._prev_window_predictions:
            return 0.5  # neutral on first window
        prev_preds = self._prev_window_predictions[config_id]
        n = min(len(prev_preds), len(current_preds))
        if n < 10:
            return 0.5
        try:
            ari = adjusted_rand_score(prev_preds[:n], current_preds[:n])
            # Map ARI [-1,1] -> [0,1]
            return (1.0 + ari) / 2.0
        except Exception:
            return 0.5

    def _penalized_score(self, base_score: float, predictions: list, expected_range=(2, 20)) -> float:
        """Fix 3: penalize configs with degenerate cluster counts."""
        n_clusters = len(set(predictions))
        if n_clusters < expected_range[0] or n_clusters > expected_range[1]:
            return base_score * 0.5
        return base_score

    def _process_window(self):
        """Evaluate all trained models on full window, pick the best."""
        if not self.current_window:
            return

        try:
            window_data = np.array(self.current_window)
            if window_data.ndim == 1:
                window_data = window_data.reshape(-1, 1)

            self.window_count += 1

            # Fix 6: store window in replay buffer (keep last 5 windows)
            self._replay_buffer.append(window_data.copy())
            if len(self._replay_buffer) > 5:
                self._replay_buffer.pop(0)

            # Fix 1: check for drift and force re-tournament if detected
            drift_detected = self.drift_detector.detect_drift(window_data)
            if drift_detected:
                self._run_tournament(window_data)
                self.optimizer.exploration_rate = min(1.0, self.optimizer.exploration_rate + 0.3)

            # Evaluate all trained models on sampled window data (fast)
            sample_size = min(200, len(window_data))
            sample_idx = np.random.choice(len(window_data), sample_size, replace=False)
            eval_data = window_data[sample_idx]

            best_config = None
            best_score = -float('inf')
            config_dicts = []
            scores = []

            new_prev_preds = {}
            for config in self.configuration_archive:
                if config.model_instance is None:
                    continue
                try:
                    predictions = [config.model_instance.predict_one(dp) for dp in eval_data]
                    valid_idx = [i for i, p in enumerate(predictions) if p >= 0]
                    if len(valid_idx) < 10:
                        continue
                    vd = eval_data[valid_idx]
                    vp = [predictions[i] for i in valid_idx]
                    if len(set(vp)) <= 1:
                        continue

                    # Fix 2: use all three CVIs properly
                    norm_scores = calculate_normalized_objective_scores(vd, vp)
                    obj = ObjectiveScore(
                        silhouette=norm_scores['normalized_silhouette'],
                        calinski=norm_scores['normalized_calinski_harabasz'],
                        davies=norm_scores['normalized_davies_bouldin']
                    )
                    composite = obj.composite_score()
                    # Fix 3: apply cluster count penalty
                    composite = self._penalized_score(composite, vp)
                    # Fix B: blend with cross-window stability (0.7 quality + 0.3 stability)
                    config_id = id(config)
                    stability = self._stability_score(config_id, vp, eval_data)
                    composite = 0.7 * composite + 0.3 * stability
                    new_prev_preds[config_id] = vp

                    if composite > 0:
                        config_dicts.append(self._config_to_dict(config))
                        scores.append(obj)
                        config.objective_scores = obj

                        if composite > best_score:
                            best_score = composite
                            best_config = config
                except Exception:
                    continue
            self._prev_window_predictions = new_prev_preds

            # Update optimizer elite
            if config_dicts and scores:
                self.optimizer.update_elite_population(config_dicts, scores)

            # Switch incumbent only if new best is meaningfully better
            if best_config is not None:
                if self.incumbent is None:
                    self.incumbent = best_config
                else:
                    incumbent_score = self._evaluate_single_config_on_data(
                        self.incumbent, eval_data)
                    if best_score > incumbent_score + 0.01:
                        self.incumbent = best_config

            # Fix 4: call adaptive rate update (was defined but never called)
            if best_score > 0:
                self._handle_adaptive_rate(best_score)
            else:
                self._previous_score = max(best_score, 0.01)

            # Fix 5: generate new configs every window (was every 3 windows)
            self._add_new_configs_trained(window_data)

            self._maintain_population_size()

        except Exception:
            pass
        finally:
            self.current_window.clear()

    def _add_new_configs_trained(self, window_data):
        """Add new configs warm-started with replay buffer + current window data."""
        # Fix 5: increased from elite_size // 3 to elite_size // 2
        n_new = max(3, self.elite_size // 2)
        for _ in range(n_new):
            try:
                cfg = self.optimizer.generate_config(self.algorithm_configs)
                if cfg and self._validate_config(cfg):
                    new_config = self._dict_to_config(cfg)
                    if new_config and new_config.model_instance is not None:
                        # Fix 6: warm-start on replay buffer before current window
                        for past_window in self._replay_buffer:
                            for dp in past_window:
                                try:
                                    new_config.model_instance.learn_one(dp)
                                except Exception:
                                    break
                        for dp in window_data:
                            try:
                                new_config.model_instance.learn_one(dp)
                            except Exception:
                                break
                        self.configuration_archive.append(new_config)
            except Exception:
                continue

    def _run_tournament(self, window_data: np.ndarray):
        """Run a tournament: generate N configs, train on first half, eval on second half."""
        half = len(window_data) // 2
        train_data = window_data[:half]
        val_data = window_data[half:]

        # Generate candidate configs
        candidates = []
        # Include configs from optimizer elite
        for _ in range(self.population_size):
            cfg = self.optimizer.generate_config(self.algorithm_configs)
            if cfg and self._validate_config(cfg):
                candidates.append(cfg)

        # Also add some pure random configs for diversity
        for _ in range(self.population_size // 2):
            cfg = self.optimizer._random_config(self.algorithm_configs)
            if self._validate_config(cfg):
                candidates.append(cfg)

        # Evaluate each candidate
        best_config = None
        best_score = -float('inf')
        best_config_dict = None
        config_dicts = []
        scores = []

        for config_dict in candidates:
            try:
                temp_config = self._dict_to_config(config_dict)
                if temp_config is None or temp_config.model_instance is None:
                    continue

                # Train on training half
                for dp in train_data:
                    try:
                        temp_config.model_instance.learn_one(dp)
                    except Exception:
                        break

                # Evaluate on validation half
                predictions = []
                for dp in val_data:
                    try:
                        predictions.append(temp_config.model_instance.predict_one(dp))
                    except Exception:
                        predictions.append(-1)

                valid_idx = [i for i, p in enumerate(predictions) if p >= 0]
                if len(valid_idx) < 10:
                    continue

                vd = val_data[valid_idx]
                vp = [predictions[i] for i in valid_idx]
                if len(set(vp)) <= 1:
                    continue

                # Score using internal CVIs
                norm_scores = calculate_normalized_objective_scores(vd, vp)
                obj = ObjectiveScore(
                    silhouette=norm_scores['normalized_silhouette'],
                    calinski=norm_scores['normalized_calinski_harabasz'],
                    davies=norm_scores['normalized_davies_bouldin']
                )
                composite = obj.composite_score()

                config_dicts.append(config_dict)
                scores.append(obj)

                if composite > best_score:
                    best_score = composite
                    best_config = temp_config
                    best_config_dict = config_dict

            except Exception:
                continue

        # Update optimizer elite with tournament results
        if config_dicts and scores:
            self.optimizer.update_elite_population(config_dicts, scores)

        # Compare tournament winner against current incumbent
        if best_config is not None:
            # Evaluate incumbent on validation data for fair comparison
            incumbent_score = 0.0
            if self.incumbent and self.incumbent.model_instance is not None:
                incumbent_score = self._evaluate_single_config_on_data(
                    self.incumbent, val_data)

            # Only switch if tournament winner is meaningfully better
            if best_score > incumbent_score + 0.02 or self.incumbent is None:
                try:
                    winner = self._dict_to_config(best_config_dict)
                    if winner and winner.model_instance is not None:
                        # Train winner on FULL window
                        for dp in window_data:
                            try:
                                winner.model_instance.learn_one(dp)
                            except Exception:
                                break
                        self.incumbent = winner
                except Exception:
                    pass

            self._previous_score = max(best_score, incumbent_score, 0.01)

    def _evaluate_single_config_on_data(self, config, data: np.ndarray) -> float:
        """Evaluate a single config's quality on given data."""
        if config is None or config.model_instance is None:
            return 0.0
        try:
            predictions = []
            for dp in data:
                try:
                    predictions.append(config.model_instance.predict_one(dp))
                except Exception:
                    predictions.append(-1)

            valid_idx = [i for i, p in enumerate(predictions) if p >= 0]
            if len(valid_idx) < 10:
                return 0.0

            vd = data[valid_idx]
            vp = [predictions[i] for i in valid_idx]
            if len(set(vp)) <= 1:
                return 0.0

            norm_scores = calculate_normalized_objective_scores(vd, vp)
            obj = ObjectiveScore(
                silhouette=norm_scores['normalized_silhouette'],
                calinski=norm_scores['normalized_calinski_harabasz'],
                davies=norm_scores['normalized_davies_bouldin']
            )
            return obj.composite_score()
        except Exception:
            return 0.0

    def _handle_adaptive_rate(self, current_score: float):
        """Adjust exploration rate based on relative improvement in composite score."""
        if self._previous_score == -float('inf'):
            # First score, just record it
            self._previous_score = current_score
            return
            
        # Calculate relative improvement
        if abs(self._previous_score) > 1e-8:  # Avoid division by zero
            relative_improvement = (current_score - self._previous_score) / abs(self._previous_score)
        else:
            relative_improvement = current_score - self._previous_score
            
        if relative_improvement > 0:
            # Score improved - decrease exploration rate (exploit more)
            adjustment_factor = min(0.5, relative_improvement)  # Cap the adjustment
            self.optimizer.exploration_rate = max(
                0.1,
                self.optimizer.exploration_rate * (1 - self.gamma * adjustment_factor)
            )
            print(f"Score improved by {relative_improvement:.4f} ({self._previous_score:.4f} -> {current_score:.4f}). Decreased exploration rate to {self.optimizer.exploration_rate:.4f}")
        else:
            # Score did not improve - increase exploration rate (explore more)
            adjustment_factor = min(0.5, abs(relative_improvement))  # Cap the adjustment
            self.optimizer.exploration_rate = min(
                1.0,
                self.optimizer.exploration_rate * (1 + self.gamma * adjustment_factor)
            )
            print(f"Score decreased by {relative_improvement:.4f} ({self._previous_score:.4f} -> {current_score:.4f}). Increased exploration rate to {self.optimizer.exploration_rate:.4f}")
            
        self._previous_score = current_score

    def _calculate_final_window_scores(self, window_data: np.ndarray) -> Tuple[List[Dict], List[ObjectiveScore]]:
        """Calculate final scores using sampled window data and direct predictions."""
        config_dicts = []
        scores = []
        # Sample data for evaluation
        sample_size = max(10, int(len(window_data) * self.sample_ratio))
        if sample_size >= len(window_data):
            sampled_data = window_data
            sample_indices = list(range(len(window_data)))
        else:
            sample_indices = np.random.choice(len(window_data), sample_size, replace=False)
            sampled_data = window_data[sample_indices]

        for config in self.configuration_archive:
            if config.model_instance is None:
                continue
            try:
                # Get predictions for sampled data using predict_one
                predictions = []
                for data_point in sampled_data:
                    try:
                        pred = config.model_instance.predict_one(data_point)
                        predictions.append(pred)
                    except Exception as e:
                        print(f"Warning: Prediction error for {config.algorithm_name}: {e}")
                        predictions.append(-1)

                # Filter out invalid predictions
                valid_indices = [i for i, pred in enumerate(predictions) if pred >= 0]
                if len(valid_indices) < 10:
                    continue

                valid_data = sampled_data[valid_indices]
                valid_predictions = [predictions[i] for i in valid_indices]

                # Check if we have multiple clusters
                unique_labels = list(set(valid_predictions))
                if len(unique_labels) <= 1:
                    continue

                # Calculate clustering validation indices
                objective_score = ObjectiveScore()
                try:
                    # Calculate normalized clustering validation indices
                    normalized_scores = calculate_normalized_objective_scores(valid_data, valid_predictions)
                    objective_score.silhouette = normalized_scores['normalized_silhouette']
                    objective_score.calinski = normalized_scores['normalized_calinski_harabasz']
                    objective_score.davies = normalized_scores['normalized_davies_bouldin']

                    # Store results
                    config_dict = self._config_to_dict(config)
                    config_dicts.append(config_dict)
                    scores.append(objective_score)
                    config.objective_scores = objective_score
                except Exception as e:
                    print(f"Warning: Error calculating scores for {config.algorithm_name}: {e}")
                    continue
            except Exception as e:
                print(f"Warning: Error evaluating {config.algorithm_name}: {e}")
                continue
        return config_dicts, scores

    def _reset_configurations_for_next_window(self):
        """Reset configuration tracking variables for the next window."""
        # Clear any temporary data if needed
        for config in self.configuration_archive:
            if hasattr(config, 'window_labels'):
                config.window_labels.clear()

    def _config_to_dict(self, config: AlgorithmConfig) -> Dict:
        """Convert AlgorithmConfig to dictionary format."""
        config_dict = {'algorithm': config.algorithm_name}
        config_dict.update(config.current_params)
        return config_dict

    def _generate_new_configs(self):
        """Generate new configurations using the optimizer."""
        new_configs = []
        for _ in range(max(1, self.elite_size)):  # Generate some new configurations
            try:
                config_dict = self.optimizer.generate_config(self.algorithm_configs)
                if config_dict and self._validate_config(config_dict):
                    new_config = self._dict_to_config(config_dict)
                    if new_config and new_config.model_instance is not None:
                        new_configs.append(new_config)
            except Exception as e:
                print(f"Warning: Error generating new configuration: {e}")
                continue

        # Add new configurations to archive
        self.configuration_archive.extend(new_configs)

    def _update_incumbent(self):
        """Update the best configuration (incumbent) based on current scores."""
        if not self.configuration_archive:
            return

        best_config = None
        best_score = -float('inf')

        # Find configuration with highest composite score
        for config in self.configuration_archive:
            try:
                if hasattr(config.objective_scores, 'composite_score'):
                    score = config.objective_scores.composite_score()
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_config = config
            except Exception as e:
                print(f"Warning: Error evaluating score for {config.algorithm_name}: {e}")
                continue

        if best_config:
            self.incumbent = best_config

    def _maintain_population_size(self):
        """Maintain archive size by keeping best and most diverse configurations."""
        if len(self.configuration_archive) <= self.population_size:
            return

        # Filter out configurations with invalid instances
        valid_configs = [cfg for cfg in self.configuration_archive
                        if cfg.model_instance is not None and
                        hasattr(cfg.objective_scores, 'composite_score')]

        if not valid_configs:
            self.configuration_archive = []
            return

        # Sort by composite score
        valid_configs.sort(key=lambda x: x.objective_scores.composite_score(), reverse=True)

        # Keep top configurations, prioritizing diversity
        selected = valid_configs[:max(1, self.population_size // 2)]
        remaining = valid_configs[max(1, self.population_size // 2):]

        # Add diverse configurations
        for cfg in remaining:
            if len(selected) < self.population_size:
                if all(self._config_distance(cfg, sel) > self.diversity_threshold for sel in selected):
                    selected.append(cfg)

        self.configuration_archive = selected[:self.population_size]

    def _config_distance(self, cfg1: AlgorithmConfig, cfg2: AlgorithmConfig) -> float:
        """Calculate distance between two configurations for diversity."""
        if cfg1.algorithm_name != cfg2.algorithm_name:
            return 1.0  # Different algorithms are maximally different

        params1 = cfg1.current_params
        params2 = cfg2.current_params

        distance = 0.0
        param_count = 0

        # Check if we have any common parameters
        common_params = set(params1.keys()) & set(params2.keys())
        if not common_params:
            return 1.0

        for key in common_params:
            param_count += 1
            if isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                # Normalize numerical differences
                diff = abs(params1[key] - params2[key])
                # Find parameter range for normalization
                for param_config in cfg1.parameters:
                    if param_config.name == key:
                        if param_config.range_min is not None and param_config.range_max is not None:
                            range_size = param_config.range_max - param_config.range_min
                            if range_size > 0:
                                diff = diff / range_size
                        break
                distance += min(1.0, diff)
            else:
                # Categorical parameters: either same (0) or different (1)
                distance += 1.0 if params1[key] != params2[key] else 0.0

        # Avoid division by zero
        if param_count == 0:
            return 1.0
            
        return distance / param_count

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary of AutoDSC."""
        if not self.configuration_archive:
            return {"error": "No configurations evaluated"}

        # Get best configuration
        best_config = self.incumbent if self.incumbent else self.configuration_archive[0]

        # Collect all composite scores
        all_scores = []
        for config in self.configuration_archive:
            try:
                if hasattr(config.objective_scores, 'composite_score'):
                    score = config.objective_scores.composite_score()
                    if np.isfinite(score):
                        all_scores.append(score)
            except:
                continue

        # Get optimizer statistics
        optimizer_stats = self.optimizer.get_exploration_stats()

        # Compile comprehensive summary
        summary = {
            "algorithm": "AutoDSC",
            "windows_processed": self.window_count,
            "configurations_evaluated": len(self.configuration_archive),
            "window_size": self.window_size,
            "exploration_rate": optimizer_stats.get("exploration_rate", None),
            "best_configuration": {
                "algorithm": best_config.algorithm_name,
                "parameters": best_config.current_params,
                "scores": {
                    "silhouette": best_config.objective_scores.silhouette,
                    "calinski": best_config.objective_scores.calinski,
                    "davies": best_config.objective_scores.davies,
                    "composite_score": (best_config.objective_scores.composite_score()
                                      if hasattr(best_config.objective_scores, 'composite_score') else 0.0)
                }
            },
            "optimizer_stats": optimizer_stats,
            "adaptive_rate_info": {
                "current_exploration_rate": self.optimizer.exploration_rate,
                "previous_best_score": self._previous_score,
                "gamma": self.gamma
            }
        }

        # Add performance statistics if scores are available
        if all_scores:
            summary["performance_statistics"] = {
                "mean_composite_score": np.mean(all_scores),
                "std_composite_score": np.std(all_scores),
                "min_composite_score": np.min(all_scores),
                "max_composite_score": np.max(all_scores),
                "num_valid_scores": len(all_scores)
            }

        return summary

    def reset(self):
        """Reset the AutoDSC to initial state."""
        # Clear all tracking data
        self.current_window.clear()
        # self.window_count = 0
        self.incumbent = None
        self._previous_score = -float('inf')

        # Reset configurations
        for config in self.configuration_archive:
            config.window_labels.clear()
            config.objective_scores = ObjectiveScore()

        # Reset optimizer
        self.optimizer.elite_population.clear()
        self.optimizer.parameter_distributions.clear()
        self.optimizer.generation_count = 0
        self.optimizer.exploration_rate = self.optimizer.exploration_rate
