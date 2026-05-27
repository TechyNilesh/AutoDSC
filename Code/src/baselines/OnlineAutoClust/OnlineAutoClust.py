import numpy as np
from stream_clustering import CluStream, DenStream, DBStream, KMeans
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import deque
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import connected_components
import warnings
warnings.filterwarnings('ignore')

class OnlineAutoClustBase:
    def __init__(self, window_size=1000, t_max=60, delta=0.02, top_n=5, ensemble_k=10):
        self.window_size = window_size
        self.t_max = t_max
        self.delta = delta
        self.top_n = top_n
        self.ensemble_k = ensemble_k
        self.cvis = ['silhouette', 'calinski', 'davies']
        self.algorithms = {
            'CluStream': self._get_clustream_space,
            'DenStream': self._get_denstream_space,
            'DBStream': self._get_dbstreak_space,
            'KMeans': self._get_kmeans_space
        }
        self.current_model = None
        self.previous_quality = None
        
    def _get_clustream_space(self):
        return {
            'algorithm': 'CluStream',
            'n_macro_clusters': hp.randint('clustream_n_macro_clusters', 2, 11),  # 2-10
            'max_micro_clusters': hp.randint('clustream_max_micro_clusters', 5, 51),  # 5-50
            'time_window': hp.randint('clustream_time_window', 100, 2001),  # 100-2000
            'micro_cluster_r_factor': hp.uniform('clustream_micro_cluster_r_factor', 1.0, 5.0),  # 1.0-5.0
            'time_gap': hp.randint('clustream_time_gap', 1, 101),  # 1-100
            'seed': hp.randint('clustream_seed', 0, 1001)
        }
    
    def _get_denstream_space(self):
        return {
            'algorithm': 'DenStream',
            'decaying_factor': hp.uniform('denstream_decaying_factor', 0.01, 0.5),
            'beta': hp.uniform('denstream_beta', 0.5, 1.0),
            'mu': hp.uniform('denstream_mu', 2.0, 10.0),
            'epsilon': hp.uniform('denstream_epsilon', 0.01, 0.5),
            'n_samples_init': hp.randint('denstream_n_samples_init', 10, 1001),
            'stream_speed': hp.randint('denstream_stream_speed', 10, 1001)
        }
    
    def _get_dbstreak_space(self):
        return {
            'algorithm': 'DBStream',
            'clustering_threshold': hp.uniform('dbstream_clustering_threshold', 0.1, 2.0),
            'fading_factor': hp.uniform('dbstream_fading_factor', 0.001, 0.1),
            'cleanup_interval': hp.randint('dbstream_cleanup_interval', 1, 101), # 1-100
            'intersection_factor': hp.uniform('dbstream_intersection_factor', 0.1, 0.9),
            'minimum_weight': hp.uniform('dbstream_minimum_weight', 0.5, 5.0)
        }
    
    def _get_kmeans_space(self):
        return {
            'algorithm': 'KMeans',
            'n_clusters': hp.randint('kmeans_n_clusters', 2, 31),  # 2-30
            'halflife': hp.uniform('kmeans_halflife', 0.01, 1.0),
            'mu': hp.uniform('kmeans_mu', 0.0, 1.0),
            'sigma': hp.uniform('kmeans_sigma', 0.1, 2.0),
            'p': hp.choice('kmeans_p', [1, 2]),
            'seed': hp.randint('kmeans_seed', 0, 1001)
        }
    
    def _create_model(self, config):
        algo = config['algorithm']
        params = {k: v for k, v in config.items() if k != 'algorithm'}
        
        try:
            if algo == 'CluStream':
                return CluStream(
                    n_macro_clusters=int(params.get('n_macro_clusters', 5)),
                    max_micro_clusters=int(params.get('max_micro_clusters', 10)),
                    time_window=int(params.get('time_window', 1000)),
                    micro_cluster_r_factor=float(params.get('micro_cluster_r_factor', 2.0)),
                    time_gap=int(params.get('time_gap', 10)),
                    seed=int(params.get('seed', 42))
                )
            elif algo == 'DenStream':
                beta = float(params.get('beta', 0.5))
                mu = float(params.get('mu', 3.0))

                # Ensure mu * beta > 1.0 constraint is satisfied
                if mu * beta <= 1.0:
                    # Adjust mu to satisfy the constraint with a small buffer
                    mu = (1.1 / beta)

                return DenStream(
                    decaying_factor=float(params.get('decaying_factor', 0.1)),
                    beta=beta,
                    mu=mu,
                    epsilon=float(params.get('epsilon', 0.02)),
                    n_samples_init=int(params.get('n_samples_init', 100)),
                    stream_speed=int(params.get('stream_speed', 100))
                )
            elif algo == 'DBStream':
                return DBStream(
                    clustering_threshold=float(params.get('clustering_threshold', 1.0)),
                    fading_factor=float(params.get('fading_factor', 0.01)),
                    cleanup_interval=float(params.get('cleanup_interval', 10.0)),
                    intersection_factor=float(params.get('intersection_factor', 0.3)),
                    minimum_weight=float(params.get('minimum_weight', 1.0))
                )
            elif algo == 'KMeans':
                return KMeans(
                    n_clusters=int(params.get('n_clusters', 5)),
                    halflife=float(params.get('halflife', 0.5)),
                    mu=float(params.get('mu', 0.0)),
                    sigma=float(params.get('sigma', 1.0)),
                    p=params.get('p', 2),
                    seed=int(params.get('seed', 42))
                )
            else:
                return None
        except Exception as e:
            print(f"Error creating {algo}: {e}")
            return None
    
    def _train_model(self, model, data):
        if model is None:
            return model

        for x in data:
            try:
                # stream_clustering algorithms expect lists
                if isinstance(x, dict):
                    x_list = [float(x[k]) for k in sorted(x.keys())]
                elif isinstance(x, (list, np.ndarray)):
                    x_list = [float(val) for val in x]
                else:
                    x_list = [float(x)]

                model.learn_one(x_list)
            except Exception as e:
                print(f"Error training model: {e}")
                continue
        return model
    
    def _predict_batch(self, model, data):
        if model is None:
            return [0] * len(data)

        predictions = []
        for x in data:
            try:
                # stream_clustering algorithms expect lists
                if isinstance(x, dict):
                    x_list = [float(x[k]) for k in sorted(x.keys())]
                elif isinstance(x, (list, np.ndarray)):
                    x_list = [float(val) for val in x]
                else:
                    x_list = [float(x)]

                pred = model.predict_one(x_list)
                predictions.append(pred)
            except Exception as e:
                predictions.append(0)  # Default prediction on error
        return predictions
    
    def _compute_cvi(self, data, labels, cvi_name):
        if len(set(labels)) < 2:
            return -10**6  # Penalize single-cluster results
        
        # Convert dict data to numpy array for sklearn metrics
        if isinstance(data[0], dict):
            data_array = np.array([[x[k] for k in sorted(x.keys())] for x in data])
        else:
            data_array = np.array(data)
        
        if cvi_name == 'silhouette':
            return silhouette_score(data_array, labels)
        elif cvi_name == 'calinski':
            return calinski_harabasz_score(data_array, labels)
        elif cvi_name == 'davies':
            return -davies_bouldin_score(data_array, labels)  # Negative since lower is better
        return 0
    
    def _ensemble_clustering(self, clusterings):
        n = len(clusterings[0])
        co_assoc = dok_matrix((n, n), dtype=np.float32)
        model_count = len(clusterings)
        
        for clustering in clusterings:
            for i in range(n):
                for j in range(i+1, n):
                    if clustering[i] == clustering[j]:
                        co_assoc[i, j] += 1
                        co_assoc[j, i] += 1
        
        co_assoc /= model_count
        graph = (co_assoc >= 0.5).astype(int).tocsr()
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        return labels
    
    def _evaluate_config(self, config, train_data, val_data, cvi_name):
        model = self._create_model(config)
        model = self._train_model(model, train_data)
        labels = self._predict_batch(model, val_data)
        score = self._compute_cvi(val_data, labels, cvi_name)
        return {'loss': -score, 'status': STATUS_OK, 'model': model, 'labels': labels}
    
    def _automl_phase(self, data):
        half = len(data) // 2
        train_data, val_data = data[:half], data[half:]
        best_models = {cvi: [] for cvi in self.cvis}
        
        for cvi in self.cvis:
            space = hp.choice('algorithm', [fn() for fn in self.algorithms.values()])
            
            def objective(config):
                return self._evaluate_config(config, train_data, val_data, cvi)
            
            trials = Trials()
            best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials, timeout=self.t_max)
            
            # Get top n models for this CVI
            sorted_trials = sorted(trials.results, key=lambda x: x['loss'])
            for trial in sorted_trials[:self.top_n]:
                best_models[cvi].append({
                    'model': trial['model'],
                    'labels': trial['labels'],
                    'score': -trial['loss']
                })
        
        # Build ensembles for each CVI
        ensembles = {}
        for cvi, models in best_models.items():
            if not models:
                continue
            clusterings = [m['labels'] for m in models]
            ensemble_labels = self._ensemble_clustering(clusterings)
            score = self._compute_cvi(val_data, ensemble_labels, cvi)
            ensembles[cvi] = {
                'models': models,
                'ensemble_labels': ensemble_labels,
                'score': score
            }
        
        # Select best ensemble
        best_cvi = max(ensembles, key=lambda k: ensembles[k]['score'])
        return best_models[best_cvi][0]['model']  # Return best model from best CVI
    
    def _detect_drift(self, current_quality):
        if self.previous_quality is None:
            return False
        return (self.previous_quality - current_quality) / self.previous_quality > self.delta
    
    def process_stream(self, data_stream):
        buffer = []
        results = []
        
        for x in data_stream:
            buffer.append(x)
            if len(buffer) >= self.window_size:
                window = buffer[-self.window_size:]
                self._process_window(window)
                results.append(self.current_model)
                buffer = []
        return results

class OnlineAutoClustVanilla(OnlineAutoClustBase):
    def _process_window(self, window):
        if self.current_model is None:
            self.current_model = self._automl_phase(window)
            labels = self._predict_batch(self.current_model, window)
            self.previous_quality = np.mean([
                self._compute_cvi(window, labels, cvi)
                for cvi in self.cvis
            ])
            return
        
        labels = self._predict_batch(self.current_model, window)
        current_quality = np.mean([
            self._compute_cvi(window, labels, cvi)
            for cvi in self.cvis
        ])
        
        if self._detect_drift(current_quality):
            self.current_model = self._automl_phase(window)
        
        self.previous_quality = current_quality

class OnlineAutoClustEnsemble(OnlineAutoClustBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup_ensemble = deque(maxlen=self.ensemble_k)
    
    def _process_window(self, window):
        if self.current_model is None:
            self.current_model = self._automl_phase(window)
            # Create a copy of the model instead of using deepcopy
            model_copy = self._copy_model(self.current_model)
            if model_copy is not None:
                self.backup_ensemble.append(model_copy)
            labels = self._predict_batch(self.current_model, window)
            self.previous_quality = np.mean([
                self._compute_cvi(window, labels, cvi)
                for cvi in self.cvis
            ])
            return
        
        labels = self._predict_batch(self.current_model, window)
        current_quality = np.mean([
            self._compute_cvi(window, labels, cvi)
            for cvi in self.cvis
        ])
        
        if self._detect_drift(current_quality):
            new_model = self._automl_phase(window)
            
            # Evaluate backup ensemble
            ensemble_labels = self._ensemble_predict(window)
            ensemble_quality = np.mean([
                self._compute_cvi(window, ensemble_labels, cvi)
                for cvi in self.cvis
            ])
            
            new_labels = self._predict_batch(new_model, window)
            new_quality = np.mean([
                self._compute_cvi(window, new_labels, cvi)
                for cvi in self.cvis
            ])
            
            if ensemble_quality > new_quality and len(self.backup_ensemble) > 0:
                self.current_model = type(new_model)()  # Placeholder for ensemble
            else:
                self.current_model = new_model
            
            self.backup_ensemble.append(new_model)
        
        self.previous_quality = current_quality
    
    def _ensemble_predict(self, data):
        # Handle empty ensemble case
        if len(self.backup_ensemble) == 0:
            return [0] * len(data)  # Default clustering
            
        all_labels = []
        for model in self.backup_ensemble:
            labels = self._predict_batch(model, data)
            all_labels.append(labels)
        return self._ensemble_clustering(all_labels)
    
    def _copy_model(self, model):
        """Create a copy of a model by creating a new instance with the same configuration"""
        if model is None:
            return None
            
        try:
            # Get the model's class name and recreate it
            model_class = type(model).__name__
            
            if model_class == 'CluStream':
                return CluStream(
                    n_macro_clusters=getattr(model, 'n_macro_clusters', 5),
                    max_micro_clusters=getattr(model, 'max_micro_clusters', 10),
                    time_window=getattr(model, 'time_window', 1000),
                    micro_cluster_r_factor=getattr(model, 'micro_cluster_r_factor', 2.0),
                    time_gap=getattr(model, 'time_gap', 10),
                    seed=getattr(model, 'seed', 42)
                )
            elif model_class == 'DenStream':
                return DenStream(
                    decaying_factor=getattr(model, 'decaying_factor', 0.1),
                    beta=getattr(model, 'beta', 0.5),
                    mu=getattr(model, 'mu', 4.0),
                    epsilon=getattr(model, 'epsilon', 0.02),
                    n_samples_init=getattr(model, 'n_samples_init', 100),
                    stream_speed=getattr(model, 'stream_speed', 100)
                )
            elif model_class == 'DBStream':
                return DBStream(
                    clustering_threshold=getattr(model, 'clustering_threshold', 1.0),
                    fading_factor=getattr(model, 'fading_factor', 0.01),
                    cleanup_interval=getattr(model, 'cleanup_interval', 10.0),
                    intersection_factor=getattr(model, 'intersection_factor', 0.3),
                    minimum_weight=getattr(model, 'minimum_weight', 1.0)
                )
            elif model_class == 'KMeans':
                return KMeans(
                    n_clusters=getattr(model, 'n_clusters', 5),
                    halflife=getattr(model, 'halflife', 0.5),
                    mu=getattr(model, 'mu', 0.0),
                    sigma=getattr(model, 'sigma', 1.0),
                    p=getattr(model, 'p', 2),
                    seed=getattr(model, 'seed', 42)
                )
            else:
                print(f"Unknown model class: {model_class}")
                return None
                
        except Exception as e:
            print(f"Error copying model: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Create data similar to CluStream example
    import random
    
    # Define base clusters with some noise
    base_data = [
        # Cluster 1 (around [1, 2])
        [1, 2], [1, 4], [1, 0], [0.8, 2.2], [1.2, 1.8], [0.9, 3.8], [1.1, 0.2],
        # Cluster 2 (around [-4, 2])
        [-4, 2], [-4, 4], [-4, 0], [-3.8, 2.1], [-4.2, 1.9], [-3.9, 3.9], [-4.1, 0.1],
        # Cluster 3 (around [5, 2])
        [5, 0], [5, 2], [5, 4], [4.8, 0.2], [5.2, 1.8], [4.9, 3.8], [5.1, 2.2]
    ]
    
    # Create a larger stream by repeating and adding noise
    stream_data = []
    for _ in range(500):  # Create 500 cycles
        for point in base_data:
            # Add small random noise to create variation
            noisy_point = {
                0: point[0] + random.uniform(-0.3, 0.3),
                1: point[1] + random.uniform(-0.3, 0.3)
            }
            stream_data.append(noisy_point)
    
    # Fix: Don't wrap in tuples - pass dictionaries directly
    data_stream = stream_data
    
    # Initialize and run Vanilla version
    vanilla = OnlineAutoClustVanilla(window_size=500, t_max=10)
    results_vanilla = vanilla.process_stream(data_stream)
    
    # Initialize and run Ensemble version
    ensemble = OnlineAutoClustEnsemble(window_size=500, t_max=10)
    results_ensemble = ensemble.process_stream(data_stream)
    
    print("Vanilla models:", len(results_vanilla))
    print("Ensemble models:", len(results_ensemble))
    print("Total data points processed:", len(stream_data))
    
    # Optional: Test predictions on known points
    if results_vanilla:
        last_model = results_vanilla[-1]
        if last_model is not None:
            try:
                print("Sample predictions from last vanilla model:")
                # Convert test points to arrays for stream_clustering algorithms
                test_point_1 = [1, 1]
                test_point_2 = [-4, 3]
                test_point_3 = [4, 3.5]
                
                print("Point [1, 1] -> Cluster:", last_model.predict_one(test_point_1))
                print("Point [-4, 3] -> Cluster:", last_model.predict_one(test_point_2))
                print("Point [4, 3.5] -> Cluster:", last_model.predict_one(test_point_3))
            except Exception as e:
                print(f"Error making predictions: {e}")
        else:
            print("No valid model found")