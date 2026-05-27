import numpy as np
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import copy
import warnings
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

@dataclass
class AlgorithmConfig:
    """Configuration for a clustering algorithm"""
    algorithm_name: str
    algorithm_class: Any
    parameters: List[ParameterConfig]
    current_params: Dict = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)

class RandomSearchStream:
    """
    Random Search algorithm for Stream Clustering using Ensemble
    Randomly samples new algorithms and configurations after each window for multiple pipelines
    """
    
    def __init__(
        self,
        ensemble_size: int = 10,
        window_size: int = 1000,
        random_seed: int = 42
    ):
        self.ensemble_size = ensemble_size
        self.window_size = window_size
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize algorithm configurations
        self.algorithm_configs = self._initialize_algorithm_configs()
        
        # Ensemble of algorithms
        self.ensemble = []
        
        # Best performer tracking
        self.incumbent = None
        self.incumbent_performance = -1
        
        # Current data window
        self.current_window = []
        self.window_count = 0
        
        # Performance tracking
        self.performance_history = []
        self.algorithm_history = []
        self.config_history = []
        
        # Initialize ensemble with random algorithms
        self._initialize_ensemble()
    
    def __str__(self):
        return "RandomSearchStream"
    
    def _initialize_algorithm_configs(self) -> List[AlgorithmConfig]:
        """Initialize available clustering algorithms with their parameter configurations using stream_clustering"""
        configs = []

        # CluStream configuration (stream_clustering version)
        clustream_params = [
            ParameterConfig("n_macro_clusters", "integer", 2, 10, default=5),
            ParameterConfig("max_micro_clusters", "integer", 5, 50, default=10),
            ParameterConfig("micro_cluster_r_factor", "numeric", 1.0, 5.0, default=2.0),
            ParameterConfig("time_window", "integer", 500, 2000, default=1000),
            ParameterConfig("time_gap", "integer", 1, 100, default=100),
            ParameterConfig("seed", "integer", 0, 1000, default=42),
        ]
        configs.append(AlgorithmConfig("CluStream", CluStream, clustream_params))

        # DenStream configuration (stream_clustering version)
        denstream_params = [
            ParameterConfig("decaying_factor", "numeric", 0.01, 0.1, default=0.01),
            ParameterConfig("beta", "numeric", 0.3, 0.8, default=0.5),
            ParameterConfig("mu", "numeric", 2.0, 5.0, default=3.0),
            ParameterConfig("epsilon", "numeric", 0.05, 0.3, default=0.1),
            ParameterConfig("n_samples_init", "integer", 50, 200, default=100),
            ParameterConfig("stream_speed", "integer", 1, 10, default=1),
        ]
        configs.append(AlgorithmConfig("DenStream", DenStream, denstream_params))

        # DBSTREAM configuration (stream_clustering version)
        dbstream_params = [
            ParameterConfig("clustering_threshold", "numeric", 0.5, 2.0, default=1.0),
            ParameterConfig("fading_factor", "numeric", 0.005, 0.05, default=0.01),
            ParameterConfig("cleanup_interval", "numeric", 500, 2000, default=1000.0),
            ParameterConfig("intersection_factor", "numeric", 0.2, 0.5, default=0.3),
            ParameterConfig("minimum_weight", "numeric", 0.5, 2.0, default=1.0),
        ]
        configs.append(AlgorithmConfig("DBSTREAM", DBStream, dbstream_params))

        # KMeans configuration (stream_clustering version)
        kmeans_params = [
            ParameterConfig("n_clusters", "integer", 2, 15, default=8),
            ParameterConfig("halflife", "numeric", 0.1, 0.9, default=0.5),
            ParameterConfig("mu", "numeric", -0.5, 0.5, default=0.0),
            ParameterConfig("sigma", "numeric", 0.5, 2.0, default=1.0),
            ParameterConfig("p", "numeric", 1.0, 3.0, default=2.0),
            ParameterConfig("seed", "integer", 0, 1000, default=42),
        ]
        configs.append(AlgorithmConfig("KMeans", KMeans, kmeans_params))

        return configs
    
    def _initialize_ensemble(self):
        """Initialize ensemble with random algorithms"""
        self.ensemble = []
        
        for i in range(self.ensemble_size):
            # Randomly select algorithm type
            selected_config = random.choice(self.algorithm_configs)
            
            # Create a copy and randomize parameters
            config_copy = copy.deepcopy(selected_config)
            
            # Randomly sample parameter values
            for param in config_copy.parameters:
                random_value = self._sample_random_parameter_value(param)
                config_copy.current_params[param.name] = random_value
                param.current_value = random_value
            
            # Create algorithm instance
            algorithm_instance = self._create_algorithm_instance(config_copy)
            
            if algorithm_instance is not None:
                self.ensemble.append({
                    'config': config_copy,
                    'algorithm': algorithm_instance,
                    'performance': 0.0,
                    'age': 0
                })
                print(f"Ensemble {i+1}: {config_copy.algorithm_name} with params: {config_copy.current_params}")
            else:
                print(f"Failed to create instance for {config_copy.algorithm_name} with params: {config_copy.current_params}")
                # Fallback to default KMeans if creation fails
                self._add_default_algorithm_to_ensemble()
    
    def _add_default_algorithm_to_ensemble(self):
        """Add default KMeans algorithm to ensemble when creation fails"""
        for config in self.algorithm_configs:
            if config.algorithm_name == "KMeans":
                config_copy = copy.deepcopy(config)
                # Set default parameters
                for param in config_copy.parameters:
                    config_copy.current_params[param.name] = param.default
                    param.current_value = param.default
                
                algorithm_instance = self._create_algorithm_instance(config_copy)
                if algorithm_instance is not None:
                    self.ensemble.append({
                        'config': config_copy,
                        'algorithm': algorithm_instance,
                        'performance': 0.0,
                        'age': 0
                    })
                    break
    
    def _sample_new_ensemble(self):
        """Randomly sample new algorithms for the entire ensemble"""
        print(f"Window {self.window_count}: Sampling new ensemble...")
        
        # Clear current ensemble
        self.ensemble = []
        
        # Create new ensemble with random algorithms
        for i in range(self.ensemble_size):
            # Randomly select algorithm type
            selected_config = random.choice(self.algorithm_configs)
            
            # Create a copy and randomize parameters
            config_copy = copy.deepcopy(selected_config)
            
            # Randomly sample parameter values
            for param in config_copy.parameters:
                random_value = self._sample_random_parameter_value(param)
                config_copy.current_params[param.name] = random_value
                param.current_value = random_value
            
            # Create algorithm instance
            algorithm_instance = self._create_algorithm_instance(config_copy)
            
            if algorithm_instance is not None:
                self.ensemble.append({
                    'config': config_copy,
                    'algorithm': algorithm_instance,
                    'performance': 0.0,
                    'age': 0
                })
            else:
                # Fallback to default algorithm
                self._add_default_algorithm_to_ensemble()
        
        print(f"New ensemble created with {len(self.ensemble)} algorithms")
    
    def _sample_random_parameter_value(self, param: ParameterConfig) -> Any:
        """Sample random parameter value based on parameter type"""
        try:
            if param.param_type == 'numeric':
                return random.uniform(param.range_min, param.range_max)
            
            elif param.param_type == 'integer':
                return random.randint(int(param.range_min), int(param.range_max))
            
            elif param.param_type == 'categorical':
                return random.choice(param.categories)
            
            elif param.param_type == 'boolean':
                return random.choice([True, False])
            
            else:
                return param.default
                
        except Exception as e:
            print(f"Error sampling parameter {param.name}: {e}")
            return param.default
    
    def _create_algorithm_instance(self, config: AlgorithmConfig):
        """Create an algorithm instance with given configuration using stream_clustering algorithms"""
        try:
            if config.algorithm_name == "CluStream":
                return config.algorithm_class(
                    n_macro_clusters=int(config.current_params.get('n_macro_clusters', 3)),
                    max_micro_clusters=int(config.current_params.get('max_micro_clusters', 5)),
                    micro_cluster_r_factor=float(config.current_params.get('micro_cluster_r_factor', 2.0)),
                    time_window=int(config.current_params.get('time_window', 1000)),
                    time_gap=int(config.current_params.get('time_gap', 100)),
                    seed=int(config.current_params.get('seed', 42))
                )
            elif config.algorithm_name == "DenStream":
                beta = float(config.current_params.get('beta', 0.5))
                mu = float(config.current_params.get('mu', 3.0))

                # Ensure mu * beta > 1.0 constraint for stream_clustering DenStream
                if mu * beta <= 1.0:
                    mu = max(mu, 1.1 / beta)

                return config.algorithm_class(
                    decaying_factor=float(config.current_params.get('decaying_factor', 0.01)),
                    beta=beta,
                    mu=mu,
                    epsilon=float(config.current_params.get('epsilon', 0.1)),
                    n_samples_init=int(config.current_params.get('n_samples_init', 100)),
                    stream_speed=int(config.current_params.get('stream_speed', 1))
                )
            elif config.algorithm_name == "DBSTREAM":
                return config.algorithm_class(
                    clustering_threshold=float(config.current_params.get('clustering_threshold', 1.0)),
                    fading_factor=float(config.current_params.get('fading_factor', 0.01)),
                    cleanup_interval=float(config.current_params.get('cleanup_interval', 1000.0)),
                    intersection_factor=float(config.current_params.get('intersection_factor', 0.3)),
                    minimum_weight=float(config.current_params.get('minimum_weight', 1.0))
                )
            elif config.algorithm_name == "KMeans":
                return config.algorithm_class(
                    n_clusters=int(config.current_params.get('n_clusters', 8)),
                    halflife=float(config.current_params.get('halflife', 0.5)),
                    mu=float(config.current_params.get('mu', 0.0)),
                    sigma=float(config.current_params.get('sigma', 1.0)),
                    p=int(config.current_params.get('p', 2))
                )
            else:
                return None
        except Exception as e:
            print(f"Error creating {config.algorithm_name}: {e}")
            return None
    
    def train(self, instance):
        """Train the algorithm with one instance"""
        # Convert instance to list format for stream_clustering
        if isinstance(instance, dict):
            # Convert dictionary to list (sorted by keys)
            x_list = [float(instance[str(i)]) for i in sorted([int(k) for k in instance.keys()])]
            x_dict = instance  # Keep dict format for silhouette calculation
        elif isinstance(instance, (list, np.ndarray)):
            x_list = [float(val) for val in instance]
            x_dict = {str(i): float(val) for i, val in enumerate(instance)}
        else:
            x_list = [float(instance)]
            x_dict = {"0": float(instance)}

        # Add to current window (store in dict format for silhouette calculation)
        self.current_window.append(x_dict)

        # Train all algorithms in ensemble
        for ensemble_item in self.ensemble:
            if ensemble_item['algorithm'] is not None:
                try:
                    ensemble_item['algorithm'].learn_one(x_list)
                    ensemble_item['age'] += 1
                except Exception as e:
                    print(f"Error in train for {ensemble_item['config'].algorithm_name}: {e}")

        # Process window when full
        if len(self.current_window) >= self.window_size:
            self._process_window()
    
    def predict_one(self, x):
        """Predict cluster for one instance using incumbent algorithm"""
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
    
    def learn_one(self, x):
        """Alias for train method to match River interface"""
        self.train(x)
    
    def _process_window(self):
        """Process the current window - evaluate performance and sample new ensemble"""
        self.window_count += 1

        if len(self.current_window) < 2:
            self.current_window = []
            return

        # Sample 25% of window size for evaluation (like AutoStreamClust)
        sample_size = max(2, int(0.25 * len(self.current_window)))
        sample_indices = np.random.choice(len(self.current_window), sample_size, replace=False)
        sample_data = [self.current_window[i] for i in sample_indices]

        # Step 1: Evaluate performance of all algorithms in ensemble using CVIs
        self._evaluate_ensemble_performance_cvi(sample_data)

        # Step 2: Select best performer as incumbent
        self._update_incumbent()

        # Step 3: Sample new ensemble for next window
        self._sample_new_ensemble()

        # Clear current window
        self.current_window = []

    def _evaluate_ensemble_performance_cvi(self, sample_data):
        """Evaluate clustering performance for all algorithms in ensemble using multiple CVIs"""
        if len(sample_data) < 2:
            return

        for ensemble_item in self.ensemble:
            cvi_score = self._calculate_cvi_score(ensemble_item['algorithm'], sample_data)
            ensemble_item['performance'] = cvi_score
            ensemble_item['config'].performance_history.append(cvi_score)

    def _calculate_cvi_score(self, algorithm, sample_data) -> float:
        """Calculate average normalized CVI score (silhouette, davies-bouldin, calinski-harabasz)"""
        try:
            if algorithm is None or len(sample_data) < 2:
                return -1.0

            # Get cluster assignments
            labels = []
            for x in sample_data:
                try:
                    x_list = [float(x[str(i)]) for i in sorted([int(k) for k in x.keys()])]
                    label = algorithm.predict_one(x_list)
                    if label is not None and not np.isnan(label):
                        labels.append(int(label))
                    else:
                        labels.append(0)
                except:
                    labels.append(0)

            unique_labels = len(set(labels))
            if unique_labels < 2:
                return -1.0

            data_array = np.array([[float(x[k]) for k in sorted(x.keys())] for x in sample_data])

            # Normalized Silhouette
            sil = silhouette_score(data_array, labels)
            norm_sil = (1 + sil) / 2

            # Normalized Davies-Bouldin (lower is better, so invert)
            db = davies_bouldin_score(data_array, labels)
            norm_db = 1.0 - np.tanh(db)

            # Normalized Calinski-Harabasz
            ch = calinski_harabasz_score(data_array, labels)
            norm_ch = np.tanh(ch / 1000.0)

            # Average normalized score
            avg_score = np.mean([norm_sil, norm_db, norm_ch])
            return avg_score

        except Exception:
            return -1.0
    
    def process_stream(self, data_stream):
        """Process a stream of data points and return results"""
        results = []
        
        for x in data_stream:
            self.train(x)
            # Store current best algorithm after each point
            if self.incumbent is not None:
                results.append(copy.deepcopy(self.incumbent))
        
        return results
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        algorithm_counts = {}
        for alg_name in self.algorithm_history:
            algorithm_counts[alg_name] = algorithm_counts.get(alg_name, 0) + 1
        
        # Get current ensemble composition
        ensemble_composition = {}
        for item in self.ensemble:
            alg_name = item['config'].algorithm_name
            ensemble_composition[alg_name] = ensemble_composition.get(alg_name, 0) + 1
        
        return {
            'windows_processed': self.window_count,
            'incumbent_algorithm': self.algorithm_history[-1] if self.algorithm_history else None,
            'current_performance': self.performance_history[-1] if self.performance_history else 0,
            'performance_history': self.performance_history,
            'algorithm_history': self.algorithm_history,
            'algorithm_usage_counts': algorithm_counts,
            'ensemble_composition': ensemble_composition,
            'ensemble_size': len(self.ensemble),
            'average_performance': np.mean(self.performance_history) if self.performance_history else 0,
            'best_performance': max(self.performance_history) if self.performance_history else 0,
            'current_ensemble_details': [
                {
                    'algorithm': item['config'].algorithm_name,
                    'params': item['config'].current_params,
                    'performance': item['performance']
                } for item in self.ensemble
            ]
        }
    
    def _update_incumbent(self):
        """Update the incumbent (best performing algorithm)"""
        best_performance = -2.0
        best_algorithm = None
        best_config = None

        for ensemble_item in self.ensemble:
            if ensemble_item['performance'] > best_performance:
                best_performance = ensemble_item['performance']
                best_algorithm = ensemble_item['algorithm']
                best_config = ensemble_item['config']

        if best_algorithm is not None:
            self.incumbent = best_algorithm
            self.incumbent_performance = best_performance
            self.performance_history.append(best_performance)
            self.algorithm_history.append(best_config.algorithm_name)
            self.config_history.append(copy.deepcopy(best_config.current_params))

            print(f"Window {self.window_count}: Best performer - {best_config.algorithm_name} with performance: {best_performance:.3f}")

            # Print ensemble performance summary
            print("Ensemble performance:")
            for i, item in enumerate(self.ensemble):
                print(f"  {i+1}. {item['config'].algorithm_name}: {item['performance']:.3f}")
            print("-" * 30)


# Example usage function
def run_random_search_experiment():
    """Run Random Search experiment using stream_clustering library"""
    
    # Import required modules for Purity Score
    from permetrics import ClusteringMetric
    
    # Initialize Random Search
    random_search = RandomSearchStream(
        ensemble_size=10,
        window_size=1000,
        random_seed=42
    )
    
    print("Random Search Stream Experiment Started")
    print("=" * 50)
    
    # Generate synthetic data for demonstration
    import random
    np.random.seed(42)
    random.seed(42)
    
    # Define base clusters with some noise and their corresponding labels
    base_clusters = [
        # Cluster 1 (around [1, 2]) - Label 0
        {"points": [[1, 2], [1, 4], [1, 0], [0.8, 2.2], [1.2, 1.8], [0.9, 3.8], [1.1, 0.2]], "label": 0},
        # Cluster 2 (around [-4, 2]) - Label 1
        {"points": [[-4, 2], [-4, 4], [-4, 0], [-3.8, 2.1], [-4.2, 1.9], [-3.9, 3.9], [-4.1, 0.1]], "label": 1},
        # Cluster 3 (around [5, 2]) - Label 2
        {"points": [[5, 0], [5, 2], [5, 4], [4.8, 0.2], [5.2, 1.8], [4.9, 3.8], [5.1, 2.2]], "label": 2}
    ]
    
    # Create a larger stream by repeating and adding noise
    stream_data = []
    true_labels = []
    n_samples = 10000
    
    for i in range(n_samples):
        # Select random cluster
        cluster = random.choice(base_clusters)
        # Select random point from cluster
        point = random.choice(cluster["points"])
        # Add small random noise to create variation
        noisy_point = {
            "0": point[0] + random.uniform(-0.3, 0.3),
            "1": point[1] + random.uniform(-0.3, 0.3)
        }
        stream_data.append(noisy_point)
        true_labels.append(cluster["label"])
    
    # Process the stream
    predicted_labels = []
    purity_scores = []
    
    for i, instance in enumerate(stream_data):
        # Train the algorithm
        random_search.train(instance)
        
        # Get prediction from current algorithm
        pred_label = random_search.predict_one(instance)
        predicted_labels.append(pred_label)
        
        # Calculate Purity Score every 1000 samples
        if (i + 1) % 1000 == 0:
            # Get the current window of true and predicted labels
            start_idx = max(0, i + 1 - 1000)
            end_idx = i + 1
            
            y_true_window = true_labels[start_idx:end_idx]
            y_pred_window = predicted_labels[start_idx:end_idx]
            
            # Calculate Purity Score using permetrics
            try:
                cm = ClusteringMetric(y_true=np.array(y_true_window), y_pred=np.array(y_pred_window))
                purity = cm.purity_score()
                purity_scores.append(purity)
                
                summary = random_search.get_performance_summary()
                print(f"Processed {i+1} samples")
                print(f"Incumbent algorithm: {summary['incumbent_algorithm']}")
                print(f"Current performance: {summary['current_performance']:.3f}")
                print(f"Purity Score: {purity:.3f}")
                print(f"Windows processed: {summary['windows_processed']}")
                print(f"Ensemble size: {summary['ensemble_size']}")
                print("-" * 30)
            except Exception as e:
                print(f"Error calculating purity score: {e}")
                purity_scores.append(0.0)
    
    # Final summary
    final_summary = random_search.get_performance_summary()
    print("\nFinal Results:")
    print("=" * 50)
    print(f"Total windows processed: {final_summary['windows_processed']}")
    print(f"Incumbent algorithm: {final_summary['incumbent_algorithm']}")
    print(f"Final performance: {final_summary['current_performance']:.3f}")
    print(f"Average performance: {final_summary['average_performance']:.3f}")
    print(f"Best performance: {final_summary['best_performance']:.3f}")
    print(f"Ensemble size: {final_summary['ensemble_size']}")
    
    # Purity Score results
    if purity_scores:
        print(f"Final Purity Score: {purity_scores[-1]:.3f}")
        print(f"Average Purity Score: {np.mean(purity_scores):.3f}")
        print(f"Best Purity Score: {max(purity_scores):.3f}")
    
    print("\nIncumbent algorithm usage statistics:")
    for alg_name, count in final_summary['algorithm_usage_counts'].items():
        percentage = (count / final_summary['windows_processed']) * 100
        print(f"  {alg_name}: {count} windows ({percentage:.1f}%)")
    
    print("\nCurrent ensemble composition:")
    for alg_name, count in final_summary['ensemble_composition'].items():
        percentage = (count / final_summary['ensemble_size']) * 100
        print(f"  {alg_name}: {count} algorithms ({percentage:.1f}%)")
    
    # Calculate overall Purity Score for the entire stream
    if len(predicted_labels) == len(true_labels):
        try:
            cm_overall = ClusteringMetric(y_true=np.array(true_labels), y_pred=np.array(predicted_labels))
            overall_purity = cm_overall.purity_score()
            print(f"\nOverall Purity Score for entire stream: {overall_purity:.3f}")
        except Exception as e:
            print(f"Error calculating overall purity score: {e}")
    
    # Test predictions on known points
    print("\nSample predictions from incumbent algorithm:")
    print("Point [1, 1] -> Cluster:", random_search.predict_one({"0": 1, "1": 1}))
    print("Point [-4, 3] -> Cluster:", random_search.predict_one({"0": -4, "1": 3}))
    print("Point [4, 3.5] -> Cluster:", random_search.predict_one({"0": 4, "1": 3.5}))
    
    # Show detailed current ensemble information
    print("\nCurrent ensemble details:")
    for i, details in enumerate(final_summary['current_ensemble_details']):
        print(f"  {i+1}. {details['algorithm']} (perf: {details['performance']:.3f}) - {details['params']}")
    
    # Return results with purity scores
    final_summary['purity_scores'] = purity_scores
    final_summary['overall_purity'] = overall_purity if 'overall_purity' in locals() else None
    
    return random_search, final_summary

if __name__ == "__main__":
    # Run the experiment
    random_search_instance, results = run_random_search_experiment()
