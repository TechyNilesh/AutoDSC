#!/usr/bin/env python3

"""
AutoDSC: Automated Data Stream Clustering Framework

Authors: Anonymous
Date: June 2024

Per-window search over a portfolio of stream-clustering algorithms, guided by
a population-based optimizer (PGSSO) and CVI-based scoring. The best-scoring
candidate each window becomes the incumbent for the next window.
"""

import numpy as np
import random
import copy
import time
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

try:
    import stream_clustering
except ImportError:
    print("Warning: stream_clustering module not found. Using fallback implementations.")



# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HyperparameterSpace:
    name: str
    param_type: str  # 'numeric', 'integer', 'categorical', 'boolean'
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    categories: Optional[List] = None
    default: Any = None


@dataclass
class CVIScore:
    """Multi-objective clustering quality score (mean of three normalized CVIs)."""
    silhouette: float = 0.0
    calinski: float = 0.0
    davies: float = 0.0

    def composite_score(self) -> float:
        return (self.silhouette + self.calinski + self.davies) / 3.0


@dataclass
class Candidate:
    algorithm_name: str
    algorithm_class: Any
    parameters: List[HyperparameterSpace]
    current_params: Dict = field(default_factory=dict)
    cvi_scores: CVIScore = field(default_factory=CVIScore)
    model_instance: Any = None


@dataclass
class EliteSolution:
    config_dict: Dict
    score: CVIScore
    frequency: int = 1
    last_seen: int = 0


# =============================================================================
# POPULATION-GUIDED STOCHASTIC SEARCH OPTIMIZER
# =============================================================================

class PGSSO:
    """Population-Guided Stochastic Search Optimizer for AutoDSC."""

    def __init__(self, elite_size: int = 5, exploration_rate: float = 0.5):
        self.elite_size = max(1, elite_size)
        self.elite_archive = []
        self.exploration_rate = max(0.1, min(1.0, exploration_rate))
        self.generation_count = 0
        self.search_distributions = {}

    def update_elite(self, configs: List[Dict], scores: List[CVIScore]):
        if not configs or not scores or len(configs) != len(scores):
            return

        new_elites = []
        for config, score in zip(configs, scores):
            if score is None or not np.isfinite(score.composite_score()):
                continue
            new_elites.append(EliteSolution(
                config_dict=copy.deepcopy(config),
                score=score,
                last_seen=self.generation_count,
            ))

        all_elites = self.elite_archive + new_elites
        all_elites.sort(
            key=lambda e: e.score.composite_score() / (1 + self.generation_count - e.last_seen),
            reverse=True,
        )
        self.elite_archive = all_elites[:self.elite_size]

        self._fit_search_distributions()
        self.generation_count += 1

    def _fit_search_distributions(self):
        self.search_distributions = {}
        if not self.elite_archive:
            return

        groups = defaultdict(lambda: defaultdict(list))
        for elite in self.elite_archive:
            algorithm_name = elite.config_dict.get('algorithm')
            if not algorithm_name:
                continue
            for param_name, value in elite.config_dict.items():
                if param_name != 'algorithm' and value is not None:
                    groups[algorithm_name][param_name].append(value)

        for algorithm_name, param_groups in groups.items():
            self.search_distributions[algorithm_name] = {}
            for param_name, values in param_groups.items():
                if not values:
                    continue
                if all(isinstance(v, (int, float)) for v in values):
                    mean_val, std_val = np.mean(values), max(0.01, np.std(values))
                    if np.isfinite(mean_val) and np.isfinite(std_val):
                        self.search_distributions[algorithm_name][param_name] = {
                            'type': 'numerical', 'mean': mean_val, 'std': std_val,
                        }
                else:
                    unique_values, counts = np.unique(values, return_counts=True)
                    self.search_distributions[algorithm_name][param_name] = {
                        'type': 'categorical',
                        'values': unique_values.tolist(),
                        'probabilities': (counts / len(values)).tolist(),
                    }

    def propose_candidate(self, search_space: List[Candidate]) -> Optional[Dict]:
        if not search_space:
            return None
        # Explore (random) vs exploit (elite-guided).
        if not self.elite_archive or random.random() < self.exploration_rate:
            return self._explore(search_space)
        return self._exploit(search_space)

    def _exploit(self, search_space: List[Candidate]) -> Dict:
        available = {c.algorithm_name for c in search_space}
        counts = defaultdict(int)
        for elite in self.elite_archive:
            name = elite.config_dict.get('algorithm')
            if name in available:
                counts[name] += 1
        if not counts:
            for c in search_space:
                counts[c.algorithm_name] = 1

        names = list(counts.keys())
        weights = np.array([counts[n] for n in names], dtype=float)
        chosen = np.random.choice(names, p=weights / weights.sum())

        candidate = next((c for c in search_space if c.algorithm_name == chosen), None)
        if not candidate:
            return self._explore(search_space)

        config_dict = {'algorithm': chosen}
        dists = self.search_distributions.get(chosen, {})
        for param in candidate.parameters:
            dist = dists.get(param.name)
            if dist is None:
                config_dict[param.name] = self._sample_parameter_randomly(param)
            elif dist['type'] == 'numerical':
                value = np.random.normal(dist['mean'], dist['std'])
                if param.range_min is not None and param.range_max is not None:
                    value = np.clip(value, param.range_min, param.range_max)
                if param.param_type == 'integer':
                    value = int(round(value))
                config_dict[param.name] = value
            else:  # categorical
                config_dict[param.name] = np.random.choice(dist['values'], p=dist['probabilities'])
        return config_dict

    def _explore(self, search_space: List[Candidate]) -> Dict:
        candidate = random.choice(search_space)
        params = {'algorithm': candidate.algorithm_name}
        for param in candidate.parameters:
            params[param.name] = self._sample_parameter_randomly(param)
        return params

    def _sample_parameter_randomly(self, param: HyperparameterSpace) -> Any:
        if param.param_type == 'integer' and param.range_min is not None and param.range_max is not None:
            return random.randint(int(param.range_min), int(param.range_max))
        if param.param_type == 'numeric' and param.range_min is not None and param.range_max is not None:
            return random.uniform(param.range_min, param.range_max)
        if param.param_type == 'categorical' and param.categories:
            return random.choice(param.categories)
        if param.param_type == 'boolean':
            return random.choice([True, False])
        return param.default

    def get_exploration_stats(self) -> Dict:
        return {
            'exploration_rate': self.exploration_rate,
            'elite_archive_size': len(self.elite_archive),
            'generation_count': self.generation_count,
            'best_score': self.elite_archive[0].score.composite_score() if self.elite_archive else 0.0,
        }


# =============================================================================
# MAIN AUTODSC CLASS
# =============================================================================

class AutoDSC:
    """
    AutoDSC: A Fast and Scalable AutoML Framework for Real-Time Stream Clustering.

    - Every window: score all incrementally trained candidates on window data.
    - Set the best-scoring candidate as the incumbent for the next window.
    - Generate new challengers via PGSSO guided by recency-weighted elite archive.
    """

    def __init__(self,
                 window_size: int = 1000,
                 population_size: int = 10,
                 random_seed: int = 42):
        self.window_size = window_size
        self.population_size = max(2, population_size)
        self.elite_size = max(1, self.population_size // 2)

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.optimizer = PGSSO(
            elite_size=self.elite_size,
            exploration_rate=0.5,
        )

        self.search_space = self._build_search_space()
        self.population = []
        self.window_buffer = []
        self.windows_processed = 0
        self.incumbent = None
        self._previous_score = -float('inf')

        # --- optional instrumentation (additive; does not change behaviour) ---
        # Per-window trace of the incumbent's internal CVI scores, the number of
        # clusters it predicted on the window, and a cumulative split of where
        # per-window wall-clock is spent. Used only by offline analysis scripts.
        self.trace = {
            'window_idx': [], 'composite': [], 'silhouette': [], 'calinski': [],
            'davies': [], 'pred_k': [], 'alpha': [],
        }
        self.timing = {'learn': 0.0, 'score': 0.0, 'search': 0.0}

        self._initialize_population()

    def __str__(self):
        return "AutoDSC"

    # ---------------------------------------------------------------- setup --

    def _build_search_space(self) -> List[Candidate]:
        short_horizon = max(10, int(0.1 * self.window_size))
        medium_horizon = max(short_horizon, int(0.5 * self.window_size))
        clustream_min_window = max(50, int(0.25 * self.window_size))
        clustream_max_window = max(clustream_min_window, int(1.5 * self.window_size))

        algorithm_specs = {
            "CluStream": {
                "class": stream_clustering.CluStream,
                "params": [
                    HyperparameterSpace("n_macro_clusters", "integer", 2, 15, default=5),
                    HyperparameterSpace("max_micro_clusters", "integer", 20, 100, default=50),
                    HyperparameterSpace(
                        "time_window", "integer",
                        clustream_min_window, clustream_max_window,
                        default=self.window_size,
                    ),
                    HyperparameterSpace("micro_cluster_r_factor", "numeric", 1.0, 3.0, default=2.0),
                ],
            },
            "DenStream": {
                "class": stream_clustering.DenStream,
                "params": [
                    HyperparameterSpace("decaying_factor", "numeric", 0.01, 0.2, default=0.1),
                    HyperparameterSpace("beta", "numeric", 0.2, 0.8, default=0.5),
                    HyperparameterSpace("mu", "numeric", 2.0, 8.0, default=4.0),
                    HyperparameterSpace("epsilon", "numeric", 0.01, 1.0, default=0.1),
                ],
            },
            "DBSTREAM": {
                "class": stream_clustering.DBStream,
                "params": [
                    HyperparameterSpace("clustering_threshold", "numeric", 0.1, 2.0, default=1.0),
                    HyperparameterSpace("fading_factor", "numeric", 0.001, 0.05, default=0.01),
                    HyperparameterSpace("intersection_factor", "numeric", 0.1, 0.8, default=0.3),
                    HyperparameterSpace("minimum_weight", "numeric", 0.5, 5.0, default=1.0),
                ],
            },
            "StreamKMeans": {
                "class": stream_clustering.StreamKMeans,
                "params": [
                    HyperparameterSpace("chunk_size", "integer", short_horizon, medium_horizon, default=200),
                    HyperparameterSpace("n_clusters", "integer", 2, 20, default=8),
                    HyperparameterSpace("halflife", "numeric", 0.05, 0.8, default=0.5),
                ],
            },
        }
        return [Candidate(name, spec["class"], spec["params"])
                for name, spec in algorithm_specs.items()]

    def _initialize_population(self):
        for _ in range(self.population_size):
            config_dict = self.optimizer._explore(self.search_space)
            if self._is_valid_config(config_dict):
                candidate = self._instantiate_candidate(config_dict)
                if candidate and candidate.model_instance is not None:
                    self.population.append(candidate)

    def _instantiate_algorithm(self, candidate: Candidate):
        denstream_init = max(10, min(500, int(0.1 * self.window_size)))
        cleanup_interval = max(10, min(200, int(0.05 * self.window_size)))
        stream_chunk = max(10, min(500, int(0.2 * self.window_size)))

        param_mapping = {
            "CluStream": {
                "n_macro_clusters": candidate.current_params.get("n_macro_clusters", 5),
                "max_micro_clusters": candidate.current_params.get("max_micro_clusters", 50),
                "time_window": candidate.current_params.get("time_window", self.window_size),
                "micro_cluster_r_factor": candidate.current_params.get("micro_cluster_r_factor", 2.0),
                "time_gap": candidate.current_params.get("time_gap", 10),
            },
            "DenStream": {
                "decaying_factor": candidate.current_params.get("decaying_factor", 0.1),
                "beta": candidate.current_params.get("beta", 0.5),
                "mu": candidate.current_params.get("mu", 4.0),
                "epsilon": candidate.current_params.get("epsilon", 0.1),
                "n_samples_init": denstream_init,
                "stream_speed": denstream_init,
            },
            "DBSTREAM": {
                "clustering_threshold": candidate.current_params.get("clustering_threshold", 1.0),
                "fading_factor": candidate.current_params.get("fading_factor", 0.01),
                "cleanup_interval": cleanup_interval,
                "intersection_factor": candidate.current_params.get("intersection_factor", 0.3),
                "minimum_weight": candidate.current_params.get("minimum_weight", 1.0),
            },
            "StreamKMeans": {
                "chunk_size": candidate.current_params.get("chunk_size", stream_chunk),
                "n_clusters": candidate.current_params.get("n_clusters", 8),
                "halflife": candidate.current_params.get("halflife", 0.5),
                "mu": 0.0,
                "sigma": 1.0,
                "p": 2.0,
            },
        }
        params = param_mapping.get(candidate.algorithm_name)
        if params is None:
            return None
        return candidate.algorithm_class(**params)

    def _instantiate_candidate(self, config_dict: Dict) -> Optional[Candidate]:
        if not config_dict or 'algorithm' not in config_dict:
            return None
        base = next((c for c in self.search_space
                     if c.algorithm_name == config_dict['algorithm']), None)
        if not base:
            return None
        candidate = Candidate(
            algorithm_name=base.algorithm_name,
            algorithm_class=base.algorithm_class,
            parameters=copy.deepcopy(base.parameters),
            current_params={k: v for k, v in config_dict.items() if k != 'algorithm'},
        )
        try:
            candidate.model_instance = self._instantiate_algorithm(candidate)
        except Exception:
            return None
        return candidate if candidate.model_instance else None

    def _is_valid_config(self, config_dict: Dict) -> bool:
        if not isinstance(config_dict, dict) or 'algorithm' not in config_dict:
            return False
        base = next((c for c in self.search_space
                     if c.algorithm_name == config_dict['algorithm']), None)
        if not base:
            return False

        for param in base.parameters:
            if param.name not in config_dict:
                continue
            value = config_dict[param.name]
            if param.param_type == 'integer':
                if isinstance(value, float) and value.is_integer():
                    config_dict[param.name] = value = int(value)
                elif not isinstance(value, int):
                    return False
            elif param.param_type == 'numeric' and not isinstance(value, (int, float)):
                return False
            elif param.param_type == 'categorical' and param.categories and value not in param.categories:
                return False
            if param.range_min is not None and value < param.range_min:
                return False
            if param.range_max is not None and value > param.range_max:
                return False
        return True

    # ------------------------------------------------------------- streaming --

    def learn_one(self, x):
        if x is None:
            raise ValueError("Input cannot be None")
        x_array = np.asarray(x).flatten()
        if x_array.size == 0:
            raise ValueError("Input array is empty")
        if not np.all(np.isfinite(x_array)):
            return

        entry = {'x': x_array.copy(), 'preds': {}}
        _t0 = time.perf_counter()
        for candidate in self.population:
            if candidate.model_instance is None:
                continue
            try:
                entry['preds'][id(candidate)] = candidate.model_instance.predict_one(x_array)
                candidate.model_instance.learn_one(x_array)
            except Exception:
                entry['preds'][id(candidate)] = None
        self.timing['learn'] += time.perf_counter() - _t0
        self.window_buffer.append(entry)

        if len(self.window_buffer) == self.window_size:
            self._window_evaluation_cycle()

    def predict_one(self, x):
        active = self.incumbent
        if not active or active.model_instance is None:
            active = next((c for c in self.population
                           if c.model_instance is not None), None)
            if active is None:
                return 0
        x_array = np.asarray(x).flatten()
        if x_array.size == 0 or not np.all(np.isfinite(x_array)):
            return 0
        try:
            return active.model_instance.predict_one(x_array)
        except Exception:
            return 0

    # --------------------------------------------------------------- scoring --

    def _compute_cvi_score(self, data: np.ndarray, predictions: List) -> Optional[CVIScore]:
        valid_idx = [i for i, p in enumerate(predictions)
                     if p is not None and p >= 0 and i < len(data)]
        if len(valid_idx) < 10:
            return None
        vd = data[valid_idx]
        vp = [predictions[i] for i in valid_idx]
        if len(set(vp)) <= 1:
            return None
        try:
            ns = {
                'sil': (1 + silhouette_score(vd, vp)) / 2,
                'ch':  np.tanh(np.log1p(calinski_harabasz_score(vd, vp)) / 10.0),
                'db':  1.0 - np.tanh(davies_bouldin_score(vd, vp)),
            }
        except Exception:
            return None
        return CVIScore(
            silhouette=float(ns['sil']),
            calinski=float(ns['ch']),
            davies=float(ns['db']),
        )

    # -------------------------------------------------------- window evaluation --

    def _window_evaluation_cycle(self):
        try:
            window_data = np.array([e['x'] for e in self.window_buffer])
            if window_data.ndim == 1:
                window_data = window_data.reshape(-1, 1)
            self.windows_processed += 1

            best_candidate, best_score = None, -float('inf')
            config_dicts, scores = [], []
            _t0 = time.perf_counter()
            for candidate in self.population:
                if candidate.model_instance is None:
                    continue
                preds = [e['preds'].get(id(candidate)) for e in self.window_buffer]
                cvi = self._compute_cvi_score(window_data, preds)
                if cvi is None:
                    continue
                config_dicts.append(self._encode_config(candidate))
                scores.append(cvi)
                candidate.cvi_scores = cvi
                composite = cvi.composite_score()
                if composite > best_score:
                    best_score, best_candidate = composite, candidate
            self.timing['score'] += time.perf_counter() - _t0

            _t1 = time.perf_counter()
            if config_dicts:
                self.optimizer.update_elite(config_dicts, scores)

            if best_candidate is not None:
                self.incumbent = best_candidate

            self._adapt_exploration_rate(best_score if best_score > 0 else 0.01)

            n_challengers = self._num_challengers()
            self._prune_population(max(1, self.population_size - n_challengers))
            self._generate_challengers(n_challengers)
            self.timing['search'] += time.perf_counter() - _t1

            self._record_trace(best_candidate)
        except Exception:
            pass
        finally:
            self.window_buffer.clear()

    def _record_trace(self, incumbent):
        """Append this window's incumbent internal scores + predicted-k (additive
        instrumentation; safe no-op on error)."""
        try:
            if incumbent is None:
                return
            cvi = incumbent.cvi_scores
            self.trace['window_idx'].append(int(self.windows_processed))
            self.trace['composite'].append(float(cvi.composite_score()))
            self.trace['silhouette'].append(float(cvi.silhouette))
            self.trace['calinski'].append(float(cvi.calinski))
            self.trace['davies'].append(float(cvi.davies))
            preds = [e['preds'].get(id(incumbent)) for e in self.window_buffer]
            self.trace['pred_k'].append(len({p for p in preds if p is not None and p >= 0}))
            self.trace['alpha'].append(float(self.optimizer.exploration_rate))
        except Exception:
            pass

    def _num_challengers(self) -> int:
        return max(1, min(max(3, self.elite_size // 2), self.population_size - 1))

    def _generate_challengers(self, n_challengers: int):
        for _ in range(n_challengers):
            cfg = self.optimizer.propose_candidate(self.search_space)
            if not (cfg and self._is_valid_config(cfg)):
                continue
            candidate = self._instantiate_candidate(cfg)
            if candidate and candidate.model_instance is not None:
                self.population.append(candidate)

    def _adapt_exploration_rate(self, current_score: float):
        if self._previous_score == -float('inf'):
            self._previous_score = current_score
            return

        if abs(self._previous_score) > 1e-8:
            relative_improvement = (current_score - self._previous_score) / abs(self._previous_score)
        else:
            relative_improvement = current_score - self._previous_score

        factor = min(0.5, abs(relative_improvement))
        if relative_improvement > 0:
            self.optimizer.exploration_rate = max(
                0.1, self.optimizer.exploration_rate * (1 - factor))
        else:
            self.optimizer.exploration_rate = min(
                1.0, self.optimizer.exploration_rate * (1 + factor))
        self._previous_score = current_score

    # ----------------------------------------------------------- population --

    def _encode_config(self, candidate: Candidate) -> Dict:
        return {'algorithm': candidate.algorithm_name, **candidate.current_params}

    def _prune_population(self, target_size: Optional[int] = None):
        target_size = self.population_size if target_size is None else max(1, target_size)
        if len(self.population) <= target_size:
            return
        valid = [c for c in self.population if c.model_instance is not None]
        if not valid:
            self.population = []
            return
        valid.sort(key=lambda c: c.cvi_scores.composite_score(), reverse=True)

        selected = []
        if self.incumbent in valid:
            selected.append(self.incumbent)

        score_quota = max(1, target_size // 2)
        for candidate in valid:
            if candidate is self.incumbent:
                continue
            if len(selected) >= score_quota:
                break
            selected.append(candidate)

        remaining = [c for c in valid if c not in selected]
        while remaining and len(selected) < target_size:
            farthest = max(
                remaining,
                key=lambda c: min(self._hyperparameter_distance(c, s) for s in selected),
            )
            selected.append(farthest)
            remaining.remove(farthest)
        self.population = selected[:target_size]

    def _hyperparameter_distance(self, c1: Candidate, c2: Candidate) -> float:
        if c1.algorithm_name != c2.algorithm_name:
            return 1.0
        common = set(c1.current_params) & set(c2.current_params)
        if not common:
            return 1.0
        ranges = {p.name: (p.range_min, p.range_max) for p in c1.parameters}
        total = 0.0
        for key in common:
            v1, v2 = c1.current_params[key], c2.current_params[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                diff = abs(v1 - v2)
                lo, hi = ranges.get(key, (None, None))
                if lo is not None and hi is not None and hi > lo:
                    diff /= (hi - lo)
                total += min(1.0, diff)
            else:
                total += 1.0 if v1 != v2 else 0.0
        return total / len(common)

    # ------------------------------------------------------------- reporting --

    def get_performance_summary(self) -> Dict:
        if not self.population:
            return {"error": "No candidates evaluated"}
        best = self.incumbent or self.population[0]
        return {
            "algorithm": "AutoDSC",
            "windows_processed": self.windows_processed,
            "population_size": len(self.population),
            "window_size": self.window_size,
            "exploration_rate": self.optimizer.exploration_rate,
            "best_candidate": {
                "algorithm": best.algorithm_name,
                "parameters": best.current_params,
                "composite_score": best.cvi_scores.composite_score(),
            },
            "optimizer_stats": self.optimizer.get_exploration_stats(),
        }

    def reset(self):
        """Reset transient state (keeps the trained population)."""
        self.window_buffer.clear()
        self.incumbent = None
        self._previous_score = -float('inf')
        for candidate in self.population:
            candidate.cvi_scores = CVIScore()
        self.optimizer.elite_archive.clear()
        self.optimizer.search_distributions.clear()
        self.optimizer.generation_count = 0
        self.optimizer.exploration_rate = 0.5
