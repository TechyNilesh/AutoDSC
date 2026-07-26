"""
Plot the per-window exploration rate (alpha_t) for AutoDSC on the Merge-Split
stream alongside the incumbent CVI score.

Output:
    charts/alpha/alpha_trace.{pdf,png}
    charts/alpha/legend.{pdf,png}
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import set_rc, clean_spines, bold_ticks, save_fig, save_legend, group_dir

HERE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(HERE, 'alpha_trace_synth_blobs_merge_split.json')
GROUP = 'alpha'
WINDOW_SIZE = 1000  # instances per window

with open(DATA) as f:
    d = json.load(f)

windows   = np.array(d['window_idx'])
alpha     = np.array(d['alpha'])
cvi       = np.array(d['composite'])

# Convert to #instances in thousands; skip first 9 windows (initial transient)
x_k    = windows * WINDOW_SIZE / 1000.0   # e.g. window 10 -> 10.0 (k)
mask   = windows >= 10
x_k, alpha, cvi = x_k[mask], alpha[mask], cvi[mask]

# Drift band boundaries in thousands of instances
MERGE_START_K, MERGE_END_K = 38.0, 52.0
SPLIT_START_K, SPLIT_END_K = 62.0, 70.0

COLOR_ALPHA = '#D55E00'   # AutoDSC vermilion-red
COLOR_CVI   = '#0072B2'   # blue

set_rc()
fig, ax1 = plt.subplots(figsize=(5.8, 3.2))

# ---- alpha (left axis) ----
ax1.plot(x_k, alpha, color=COLOR_ALPHA, linewidth=2.6)
ax1.set_xlabel('#Instances (in thousands)', fontweight='bold', fontsize=12)
ax1.set_ylabel(r'Exploration Rate $\alpha_t$', color=COLOR_ALPHA,
               fontweight='bold', fontsize=11)
ax1.tick_params(axis='y', labelcolor=COLOR_ALPHA)
ax1.set_xlim(10, 100)
ax1.set_ylim(0.22, 0.30)   # comfortable padding around [0.244, 0.268]

# ---- CVI (right axis) ----
ax2 = ax1.twinx()
ax2.plot(x_k, cvi, color=COLOR_CVI, linewidth=1.8, linestyle='--')
ax2.set_ylabel('Incumbent CVI Score', color=COLOR_CVI,
               fontweight='bold', fontsize=11)
ax2.tick_params(axis='y', labelcolor=COLOR_CVI)
ax2.set_ylim(0.70, 0.90)   # comfortable padding around [0.762, 0.836]

# ---- drift bands ----
for start, end, label in [
    (MERGE_START_K, MERGE_END_K, 'Merge'),
    (SPLIT_START_K, SPLIT_END_K, 'Split'),
]:
    ax1.axvspan(start, end, alpha=0.12, color='grey', zorder=0)
    ax1.text((start + end) / 2, 0.222,
             label, ha='center', va='bottom', fontsize=9,
             fontweight='bold', color='#444444')

clean_spines(ax1)
bold_ticks(ax1)
bold_ticks(ax2)

fig.tight_layout()
save_fig(fig, GROUP, 'alpha_trace')

# ---- standalone legend ----
handles = [
    Line2D([0], [0], color=COLOR_ALPHA, linewidth=2.6,
           label=r'Exploration Rate $\alpha_t$'),
    Line2D([0], [0], color=COLOR_CVI, linewidth=1.8, linestyle='--',
           label='Incumbent CVI Score'),
]
save_legend(handles, GROUP, name='legend', ncol=2, width=5.5)

print(f"Saved -> charts/{GROUP}/alpha_trace.{{pdf,png}} + legend.{{pdf,png}}")
