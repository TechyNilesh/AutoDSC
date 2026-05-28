import numpy as np
import random
import copy
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from sklearn.metrics import silhouette_score
from collections import defaultdict, deque
warnings.filterwarnings('ignore')

# Import stream_clustering algorithms
from stream_clustering import CluStream, DenStream, DBStream, KMeans

@dataclass
class ParameterConfig:
    """Configuration for a parameter with its type and range"""
    name: str
    param_type: str  # 'numeric', 'integer', 'categorical', 'boolean'
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    categories: Optional[List] = None
    default: Any = None
    current_value: Any = None
    # For probability distributions
    std_dev: Optional[float] = None
    probabilities: Optional[Dict] = None

@dataclass
class AlgorithmConfig:
    """Configuration for a clustering algorithm"""
    algorithm_name: str
    algorithm_class: Any
    parameters: List[ParameterConfig]
    current_params: Dict = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)

class ConfStream:
    """
    Exact implementation of confStream as described in the paper:
    "confStream: Automated Algorithm Selection and Configuration of Stream Clustering Algorithms"
    """
    
    def __init__(
        self,
        ensemble_size: int = 20,
        window_size: int = 1000,
        fading_factor: float = 0.05,
        reset_probability: float = 0.01,
        test_ensemble_size: int = 10,
        random_seed: int = 42
    ):
        self.ensemble_size = ensemble_size
        self.window_size = window_size
        self.fading_factor = fading_factor
        self.reset_probability = reset_probability
        self.test_ensemble_size = test_ensemble_size
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize algorithm configurations
        self.algorithm_configs = self._initialize_algorithm_configs()
        
        # Initialize ensembles
        self.active_ensemble = []
        self.test_ensemble = []
        self.test_ensemble_ready = []  # Test ensemble from previous window
        
        # Current data windows
        self.current_window = []
        self.window_count = 0
        
        # Best performer tracking
        self.incumbent = None
        self.incumbent_performance = -1
        
        # Performance tracking
        self.performance_history = []
        
        self._initialize_ensembles()
    
    def __str__(self):
        return "ConfStream"
    
    def _initialize_algorithm_configs(self) -> List[AlgorithmConfig]:
        """Initialize available clustering algorithms with their parameter configurations using River"""
        configs = []
        
        # CluStream configuration (stream_clustering version)
        clustream_params = [
            ParameterConfig("n_macro_clusters", "integer", 2, 10, default=5),
            ParameterConfig("max_micro_clusters", "integer", 5, 50, default=10),
            ParameterConfig("micro_cluster_r_factor", "numeric", 1.0, 5.0, default=2.0),
            ParameterConfig("time_window", "integer", 100, 2000, default=1000),
            ParameterConfig("time_gap", "integer", 1, 100, default=10),
            ParameterConfig("seed", "integer", 0, 1000, default=42),
        ]
        configs.append(AlgorithmConfig("CluStream", CluStream, clustream_params))
        
        # DenStream configuration (stream_clustering version)
        denstream_params = [
            ParameterConfig("decaying_factor", "numeric", 0.01, 0.5, default=0.1),
            ParameterConfig("beta", "numeric", 0.5, 1.0, default=0.5),
            ParameterConfig("mu", "numeric", 2.0, 10.0, default=4.0),
            ParameterConfig("epsilon", "numeric", 0.01, 0.5, default=0.02),
            ParameterConfig("n_samples_init", "integer", 10, 1000, default=100),
            ParameterConfig("stream_speed", "integer", 10, 1000, default=100),
        ]
        configs.append(AlgorithmConfig("DenStream", DenStream, denstream_params))
        
        # DBSTREAM configuration (stream_clustering version)
        dbstream_params = [
            ParameterConfig("clustering_threshold", "numeric", 0.1, 2.0, default=1.0),
            ParameterConfig("fading_factor", "numeric", 0.001, 0.1, default=0.01),
            ParameterConfig("cleanup_interval", "integer", 1, 100, default=10),
            ParameterConfig("intersection_factor", "numeric", 0.1, 0.9, default=0.3),
            ParameterConfig("minimum_weight", "numeric", 0.5, 5.0, default=1.0),
        ]
        configs.append(AlgorithmConfig("DBSTREAM", DBStream, dbstream_params))
        
        # KMeans configuration (stream_clustering version)
        kmeans_params = [
            ParameterConfig("n_clusters", "integer", 2, 30, default=5),
            ParameterConfig("halflife", "numeric", 0.01, 1.0, default=0.5),
            ParameterConfig("mu", "numeric", 0.0, 1.0, default=0.0),
            ParameterConfig("sigma", "numeric", 0.1, 2.0, default=1.0),
            ParameterConfig("p", "categorical", categories=[1, 2], default=2),
            ParameterConfig("seed", "integer", 0, 1000, default=42),
        ]
        configs.append(AlgorithmConfig("KMeans", KMeans, kmeans_params))

        return configs
    
    def _initialize_ensembles(self):
        """Initialize active ensemble with default configurations"""
        for config in self.algorithm_configs:
            # Create default configuration
            default_params = {}
            for param in config.parameters:
                default_params[param.name] = param.default
                param.current_value = param.default
                # Initialize probability distributions
                if param.param_type == 'numeric':
                    param.std_dev = (param.range_max - param.range_min) / 2
                elif param.param_type == 'integer':
                    param.std_dev = (param.range_max - param.range_min) / 2
                elif param.param_type == 'categorical' and param.categories:
                    param.probabilities = {cat: 1.0/len(param.categories) 
                                         for cat in param.categories}
                elif param.param_type == 'boolean':
                    param.probabilities = {True: 0.5, False: 0.5}
            
            config.current_params = default_params
            
            # Create algorithm instance
            algorithm_instance = self._create_algorithm_instance(config)
            
            if algorithm_instance is not None:
                self.active_ensemble.append({
                    'config': copy.deepcopy(config),
                    'algorithm': algorithm_instance,
                    'performance': 0.0,
                    'age': 0
                })
    
    def _create_algorithm_instance(self, config: AlgorithmConfig):
        """Create an algorithm instance with given configuration using River algorithms"""
        try:
            if config.algorithm_name == "CluStream":
                return config.algorithm_class(
                    n_macro_clusters=int(config.current_params.get('n_macro_clusters', 5)),
                    max_micro_clusters=int(config.current_params.get('max_micro_clusters', 100)),
                    micro_cluster_r_factor=float(config.current_params.get('micro_cluster_r_factor', 2.0)),
                    time_window=int(config.current_params.get('time_window', 1000)),
                    time_gap=int(config.current_params.get('time_gap', 100)),
                    seed=int(config.current_params.get('seed', 42))
                )
            elif config.algorithm_name == "DenStream":
                beta = float(config.current_params.get('beta', 0.75))
                mu = float(config.current_params.get('mu', 2.0))
                
                # Ensure mu * beta > 1.0 constraint for River DenStream
                if mu * beta <= 1.0:
                    mu = max(mu, 1.1 / beta)
                
                return config.algorithm_class(
                    decaying_factor=float(config.current_params.get('decaying_factor', 0.25)),
                    beta=beta,
                    mu=mu,
                    epsilon=float(config.current_params.get('epsilon', 0.02)),
                    n_samples_init=int(config.current_params.get('n_samples_init', 1000)),
                    stream_speed=int(config.current_params.get('stream_speed', 100))
                )
            elif config.algorithm_name == "DBSTREAM":
                return config.algorithm_class(
                    clustering_threshold=float(config.current_params.get('clustering_threshold', 1.0)),
                    fading_factor=float(config.current_params.get('fading_factor', 0.01)),
                    cleanup_interval=float(config.current_params.get('cleanup_interval', 2)),
                    intersection_factor=float(config.current_params.get('intersection_factor', 0.3)),
                    minimum_weight=float(config.current_params.get('minimum_weight', 1.0))
                )
            elif config.algorithm_name == "KMeans":
                return config.algorithm_class(
                    n_clusters=int(config.current_params.get('n_clusters', 5)),
                    halflife=float(config.current_params.get('halflife', 0.5)),
                    mu=float(config.current_params.get('mu', 0.0)),
                    sigma=float(config.current_params.get('sigma', 1.0)),
                    p=config.current_params.get('p', 2)
                )
            else:
                return None
        except Exception as e:
            print(f"Error creating {config.algorithm_name}: {e}")
            return None
    
    def learn_one(self, x):
        """Learn from one instance - main interface"""
        # Convert instance to list format for stream_clustering
        if isinstance(x, dict):
            # Convert dictionary to list (sorted by keys)
            x_list = [float(x[str(i)]) for i in sorted([int(k) for k in x.keys()])]
            x_dict = x  # Keep dict format for silhouette calculation
        elif isinstance(x, (list, np.ndarray)):
            x_list = [float(val) for val in x]
            x_dict = {str(i): float(val) for i, val in enumerate(x)}
        else:
            x_list = [float(x)]
            x_dict = {"0": float(x)}

        # Add to current window (store in dict format for silhouette calculation)
        self.current_window.append(x_dict)

        # Train all algorithms in active ensemble
        for ensemble_item in self.active_ensemble:
            if ensemble_item['algorithm'] is not None:
                try:
                    ensemble_item['algorithm'].learn_one(x_list)
                    ensemble_item['age'] += 1
                except Exception:
                    pass

        # Train test ensemble ready for comparison (from previous window)
        for ensemble_item in self.test_ensemble_ready:
            if ensemble_item['algorithm'] is not None:
                try:
                    ensemble_item['algorithm'].learn_one(x_list)
                    ensemble_item['age'] += 1
                except Exception:
                    pass
        
        # Process window when full
        if len(self.current_window) >= self.window_size:
            self._process_window()
    
    def predict_one(self, x):
        """Predict cluster for one instance"""
        if self.incumbent is not None:
            try:
                # Convert to list format for stream_clustering
                if isinstance(x, dict):
                    x_list = [float(x[str(i)]) for i in sorted([int(k) for k in x.keys()])]
                elif isinstance(x, (list, np.ndarray)):
                    x_list = [float(val) for val in x]
                else:
                    x_list = [float(x)]

                return self.incumbent.predict_one(x_list)
            except:
                return 0
        return 0
    
    def _process_window(self):
        """Process the current window according to paper algorithm"""
        self.window_count += 1
        
        if len(self.current_window) < 2:
            self.current_window = []
            return
        
        # Step 1: Evaluate performance of all algorithms in active ensemble
        self._evaluate_active_ensemble()
        
        # Step 2: Evaluate test ensemble from previous window (if exists)
        if self.test_ensemble_ready:
            self._evaluate_test_ensemble()
            
            # Step 3: Update ensemble composition based on test results
            self._update_ensemble_composition()
        
        # Step 4: Select best performer as incumbent
        self._update_incumbent()
        
        # Step 5: Generate new test ensemble for next window
        self._generate_test_ensemble()
        
        # Clear current window
        self.current_window = []
    
    def _evaluate_active_ensemble(self):
        """Evaluate clustering performance of active ensemble using Silhouette Width"""
        if len(self.current_window) < 2:
            return
        
        for ensemble_item in self.active_ensemble:
            silhouette_width = self._calculate_silhouette_width(ensemble_item['algorithm'])
            ensemble_item['performance'] = silhouette_width
            ensemble_item['config'].performance_history.append(silhouette_width)
    
    def _evaluate_test_ensemble(self):
        """Evaluate clustering performance of test ensemble using Silhouette Width"""
        if len(self.current_window) < 2:
            return
        
        for ensemble_item in self.test_ensemble_ready:
            silhouette_width = self._calculate_silhouette_width(ensemble_item['algorithm'])
            ensemble_item['performance'] = silhouette_width
    
    def _calculate_silhouette_width(self, algorithm) -> float:
        """Calculate Silhouette Width as per paper equation (1)"""
        try:
            if algorithm is None or len(self.current_window) < 2:
                return -1.0
            
            # Get cluster assignments
            labels = []
            for x in self.current_window:
                try:
                    # Convert dict to list for stream_clustering
                    x_list = [float(x[str(i)]) for i in sorted([int(k) for k in x.keys()])]
                    label = algorithm.predict_one(x_list)

                    if label is not None and not np.isnan(label):
                        labels.append(int(label))
                    else:
                        labels.append(0)
                except:
                    labels.append(0)
            
            # Check if we have at least 2 clusters
            unique_labels = len(set(labels))
            if unique_labels < 2:
                return -1.0
            
            # Convert data to numpy array (extract numeric values in sorted order)
            data_array = np.array([[float(x[k]) for k in sorted(x.keys())] for x in self.current_window])
            
            # Calculate Silhouette Width
            return silhouette_score(data_array, labels)
            
        except Exception as e:
            return -1.0
    
    def _update_incumbent(self):
        """Update the incumbent (best performing algorithm)"""
        best_performance = -2.0
        best_algorithm = None
        
        for ensemble_item in self.active_ensemble:
            if ensemble_item['performance'] > best_performance:
                best_performance = ensemble_item['performance']
                best_algorithm = ensemble_item['algorithm']
        
        if best_algorithm is not None:
            self.incumbent = best_algorithm
            self.incumbent_performance = best_performance
            self.performance_history.append(best_performance)
    
    def _generate_test_ensemble(self):
        """Generate new test ensemble for next window (Step 3 in paper)"""
        # Move current test ensemble to ready state
        self.test_ensemble_ready = self.test_ensemble
        self.test_ensemble = []
        
        # Generate new configurations for test ensemble
        for _ in range(self.test_ensemble_size):
            # Step 2: Sample parent configuration
            parent = self._sample_parent_configuration()
            if parent is None:
                continue
            
            # Step 3: Create offspring configuration
            offspring_config = self._create_offspring_configuration(parent['config'])
            if offspring_config is None:
                continue
            
            # Create algorithm instance
            algorithm_instance = self._create_algorithm_instance(offspring_config)
            if algorithm_instance is None:
                continue
            
            # Add to test ensemble
            self.test_ensemble.append({
                'config': offspring_config,
                'algorithm': algorithm_instance,
                'performance': 0.0,
                'age': 0
            })
    
    def _sample_parent_configuration(self) -> Optional[Dict]:
        """Sample parent configuration using roulette wheel selection"""
        # Get performance values and shift them to handle negative values
        performances = [item['performance'] for item in self.active_ensemble]
        
        if not performances:
            return None
        
        # Shift negative values as per paper: |min x| - min x
        min_perf = min(performances)
        shifted_performances = [abs(min_perf) - min_perf + p for p in performances]
        total_fitness = sum(shifted_performances)
        
        if total_fitness <= 0:
            return random.choice(self.active_ensemble)
        
        # Roulette wheel selection
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        
        for i, fitness in enumerate(shifted_performances):
            current_sum += fitness
            if current_sum >= selection_point:
                return self.active_ensemble[i]
        
        return self.active_ensemble[-1]
    
    def _create_offspring_configuration(self, parent_config: AlgorithmConfig) -> Optional[AlgorithmConfig]:
        """Create offspring configuration from parent using paper's method"""
        try:
            offspring_config = copy.deepcopy(parent_config)
            
            # Reset parameters with probability p (exploration vs exploitation)
            if random.random() < self.reset_probability:
                self._reset_parameter_distributions(offspring_config)
            
            # Sample new parameter values
            for param in offspring_config.parameters:
                new_value = self._sample_parameter_value(param)
                offspring_config.current_params[param.name] = new_value
                param.current_value = new_value
                
                # Update parameter distribution (exploitation)
                self._update_parameter_distribution(param)
            
            return offspring_config
        except Exception as e:
            return None
    
    def _sample_parameter_value(self, param: ParameterConfig) -> Any:
        """Sample new parameter value according to paper's method"""
        try:
            if param.param_type == 'numeric':
                # Truncated normal distribution N(μ, σ) with μ at parent position
                mean = param.current_value
                std = param.std_dev if param.std_dev else (param.range_max - param.range_min) / 2
                
                # Sample from truncated normal
                for _ in range(100):  # Max attempts
                    value = np.random.normal(mean, std)
                    if param.range_min <= value <= param.range_max:
                        return value
                
                # Fallback to uniform sampling
                return random.uniform(param.range_min, param.range_max)
            
            elif param.param_type == 'integer':
                # Same as numeric but rounded
                mean = param.current_value
                std = param.std_dev if param.std_dev else (param.range_max - param.range_min) / 2
                
                for _ in range(100):
                    value = int(round(np.random.normal(mean, std)))
                    if param.range_min <= value <= param.range_max:
                        return value
                
                return random.randint(int(param.range_min), int(param.range_max))
            
            elif param.param_type in ['categorical', 'boolean']:
                # Sample according to probability list
                if param.probabilities:
                    categories = list(param.probabilities.keys())
                    probabilities = list(param.probabilities.values())
                    return np.random.choice(categories, p=probabilities)
                else:
                    return random.choice(param.categories if param.categories else [True, False])
            
            else:
                return param.default
                
        except Exception as e:
            return param.default
    
    def _update_parameter_distribution(self, param: ParameterConfig):
        """Update parameter distribution according to paper's equations"""
        if param.param_type in ['numeric', 'integer']:
            # Exponential decay: σt+1 = σt · 2^(-λ)
            if param.std_dev:
                param.std_dev *= (2 ** (-self.fading_factor))
        
        elif param.param_type in ['categorical', 'boolean'] and param.probabilities:
            # Increase probability: pt+1 = pt · (2 - 2^(-λ))
            current_val = param.current_value
            if current_val in param.probabilities:
                factor = 2 - (2 ** (-self.fading_factor))
                param.probabilities[current_val] *= factor
                
                # Normalize probabilities
                total = sum(param.probabilities.values())
                if total > 0:
                    for key in param.probabilities:
                        param.probabilities[key] /= total
    
    def _reset_parameter_distributions(self, config: AlgorithmConfig):
        """Reset parameter distributions for exploration"""
        for param in config.parameters:
            if param.param_type in ['numeric', 'integer']:
                param.std_dev = (param.range_max - param.range_min) / 2
            elif param.param_type == 'categorical' and param.categories:
                param.probabilities = {cat: 1.0/len(param.categories) 
                                     for cat in param.categories}
            elif param.param_type == 'boolean':
                param.probabilities = {True: 0.5, False: 0.5}
    
    def _update_ensemble_composition(self):
        """Update ensemble composition by replacing poor performers (Step 5 in paper)"""
        if not self.test_ensemble_ready:
            return
        
        # Compare test ensemble with active ensemble
        # Sort by performance for replacement
        self.test_ensemble_ready.sort(key=lambda x: x['performance'], reverse=True)
        self.active_ensemble.sort(key=lambda x: x['performance'])
        
        # Replace worst performers with best test configurations
        replacements_made = 0
        
        for i, test_config in enumerate(self.test_ensemble_ready):
            if i >= len(self.active_ensemble):
                break
            
            test_perf = test_config['performance']
            active_perf = self.active_ensemble[i]['performance']
            
            # Replace if test configuration outperforms active configuration
            if test_perf > active_perf:
                # Don't replace the best configuration of each algorithm type
                algorithm_name = self.active_ensemble[i]['config'].algorithm_name
                if not self._is_best_of_algorithm_type(i, algorithm_name):
                    # Create new algorithm instance for replacement
                    new_config = copy.deepcopy(test_config['config'])
                    new_algorithm = self._create_algorithm_instance(new_config)
                    
                    if new_algorithm is not None:
                        self.active_ensemble[i] = {
                            'config': new_config,
                            'algorithm': new_algorithm,
                            'performance': test_perf,
                            'age': 0
                        }
                        replacements_made += 1
        
        # Clear test ensemble ready
        self.test_ensemble_ready = []
        
        if replacements_made > 0:
            print(f"Window {self.window_count}: Made {replacements_made} replacements")
    
    def _is_best_of_algorithm_type(self, index: int, algorithm_name: str) -> bool:
        """Check if this is the best performer of its algorithm type"""
        best_performance = -2.0
        best_index = -1
        
        for i, item in enumerate(self.active_ensemble):
            if item['config'].algorithm_name == algorithm_name:
                if item['performance'] > best_performance:
                    best_performance = item['performance']
                    best_index = i
        
        return index == best_index
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'windows_processed': self.window_count,
            'current_performance': self.incumbent_performance,
            'performance_history': self.performance_history,
            'ensemble_composition': {
                item['config'].algorithm_name: item['performance'] 
                for item in self.active_ensemble
            },
            'algorithm_counts': {
                alg_name: sum(1 for item in self.active_ensemble 
                             if item['config'].algorithm_name == alg_name)
                for alg_name in set(item['config'].algorithm_name for item in self.active_ensemble)
            }
        }


# Example usage function
def run_confstream_paper_experiment():
    """Run ConfStream experiment exactly as described in the paper"""
    
    # Initialize ConfStream with paper parameters
    conf_stream = ConfStream(
        ensemble_size=20,        # esize = 20
        window_size=1000,        # h = 1000
        fading_factor=0.05,      # λ = 0.05
        reset_probability=0.01,  # p = 0.01
        test_ensemble_size=10,   # etest = 10
        random_seed=42
    )
    
    print("ConfStream Paper Implementation Experiment")
    print("=" * 50)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    random.seed(42)
    
    # Create 3 well-separated clusters
    n_samples = 10000
    data_stream = []
    true_labels = []
    
    for i in range(n_samples):
        # Generate 3 clusters
        cluster_id = i % 3
        if cluster_id == 0:
            # Cluster 1: center at (0, 0)
            point = {"0": np.random.normal(0, 0.5), "1": np.random.normal(0, 0.5)}
        elif cluster_id == 1:
            # Cluster 2: center at (3, 3)
            point = {"0": np.random.normal(3, 0.5), "1": np.random.normal(3, 0.5)}
        else:
            # Cluster 3: center at (-3, 3)
            point = {"0": np.random.normal(-3, 0.5), "1": np.random.normal(3, 0.5)}
        
        data_stream.append(point)
        true_labels.append(cluster_id)
    
    # Process the stream
    for i, instance in enumerate(data_stream):
        conf_stream.learn_one(instance)
        
        # Print progress every 1000 samples
        if (i + 1) % 1000 == 0:
            summary = conf_stream.get_performance_summary()
            print(f"Processed {i+1} samples")
            print(f"Windows processed: {summary['windows_processed']}")
            print(f"Current performance (Silhouette): {summary['current_performance']:.3f}")
            print(f"Algorithm counts: {summary['algorithm_counts']}")
            print("-" * 30)
    
    # Final summary
    final_summary = conf_stream.get_performance_summary()
    print("\nFinal Results:")
    print("=" * 50)
    print(f"Total windows processed: {final_summary['windows_processed']}")
    print(f"Final performance (Silhouette): {final_summary['current_performance']:.3f}")
    if final_summary['performance_history']:
        print(f"Average performance: {np.mean(final_summary['performance_history']):.3f}")
        print(f"Best performance: {max(final_summary['performance_history']):.3f}")
    
    print("\nFinal ensemble composition:")
    for alg_name, perf in final_summary['ensemble_composition'].items():
        print(f"  {alg_name}: {perf:.3f}")
    
    print("\nAlgorithm distribution in final ensemble:")
    for alg_name, count in final_summary['algorithm_counts'].items():
        print(f"  {alg_name}: {count} configurations")
    
    # Test predictions
    if conf_stream.incumbent is not None:
        print("\nSample predictions from incumbent:")
        print("Point (0, 0) -> Cluster:", conf_stream.predict_one({"0": 0, "1": 0}))
        print("Point (3, 3) -> Cluster:", conf_stream.predict_one({"0": 3, "1": 3}))
        print("Point (-3, 3) -> Cluster:", conf_stream.predict_one({"0": -3, "1": 3}))
    
    return conf_stream, final_summary

if __name__ == "__main__":
    # Run the experiment
    confstream_instance, results = run_confstream_paper_experiment()
