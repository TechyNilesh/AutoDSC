#!/usr/bin/env python3

"""
AutoDSC Ablation Variants

Leave-one-out ablation of AutoDSC's key components. Implemented entirely as
subclasses so that AutoDSC.py is never modified — the full algorithm and the
ablated variants share exactly the same code path except for the single
component under study.

Variants:
    full         complete AutoDSC (sanity check, identical to AutoDSC)
    no_recency   raw-score elite ranking (drop recency weighting)
    no_adaptive  fixed exploration rate (drop adaptive explore/exploit)
    single_cvi   Silhouette-only scoring (drop the 3-CVI composite)
    no_diversity score-only top-N pruning (drop diversity preservation)

Note: pure random search (no elite guidance) is intentionally NOT included
here because it is already covered by the RandomSearchStream baseline.
"""

import copy
import numpy as np
from typing import Dict, List, Optional

from AutoDSC import AutoDSC, PGSSO, CVIScore, Candidate, EliteSolution


ABLATIONS = ['full', 'no_recency', 'no_adaptive', 'single_cvi', 'no_diversity']


class AblationPGSSO(PGSSO):
    """PGSSO variant that can drop the recency weighting of the elite archive."""

    def __init__(self, *args, use_recency: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_recency = use_recency

    def update_elite(self, configs: List[Dict], scores: List[CVIScore]):
        if self.use_recency:
            return super().update_elite(configs, scores)

        # --- no_recency: rank elites by raw composite score (no aging) ---
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
        all_elites.sort(key=lambda e: e.score.composite_score(), reverse=True)
        self.elite_archive = all_elites[:self.elite_size]
        self._fit_search_distributions()
        self.generation_count += 1


class AutoDSCAblation(AutoDSC):
    """AutoDSC with a single component ablated, selected by `ablation`."""

    def __init__(self, *args, ablation: str = 'full', **kwargs):
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation '{ablation}'. Choose from {ABLATIONS}.")
        self._ablation = ablation
        super().__init__(*args, **kwargs)

        # Swap in the recency-aware optimizer. The base __init__ already built
        # the initial population via _explore (drift-independent), and the
        # elite archive is empty at this point, so no state is lost.
        opt = AblationPGSSO(
            elite_size=self.elite_size,
            exploration_rate=0.5,
            use_recency=(ablation != 'no_recency'),
        )
        self.optimizer = opt

    def __str__(self):
        return f"AutoDSC_Ablation_{self._ablation}"

    # --- single_cvi: collapse the 3-CVI composite to Silhouette only ---
    def _compute_cvi_score(self, data: np.ndarray, predictions: List) -> Optional[CVIScore]:
        cvi = super()._compute_cvi_score(data, predictions)
        if cvi is None or self._ablation != 'single_cvi':
            return cvi
        # Make composite == silhouette by setting all three fields equal.
        return CVIScore(silhouette=cvi.silhouette,
                        calinski=cvi.silhouette,
                        davies=cvi.silhouette)

    # --- no_adaptive: keep exploration rate fixed at its initial value ---
    def _adapt_exploration_rate(self, current_score: float):
        if self._ablation == 'no_adaptive':
            return
        super()._adapt_exploration_rate(current_score)

    # --- no_diversity: pure score-only top-N pruning ---
    def _prune_population(self, target_size: Optional[int] = None):
        if self._ablation != 'no_diversity':
            return super()._prune_population(target_size)
        target_size = self.population_size if target_size is None else max(1, target_size)
        if len(self.population) <= target_size:
            return
        valid = [c for c in self.population if c.model_instance is not None]
        if not valid:
            self.population = []
            return
        valid.sort(key=lambda c: c.cvi_scores.composite_score(), reverse=True)
        self.population = valid[:target_size]
