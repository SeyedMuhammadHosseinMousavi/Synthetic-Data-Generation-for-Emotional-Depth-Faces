import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Configure fonts and style
# ------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 16,
    'font.weight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# ------------------------------
# Prepare the data
# ------------------------------
methods = ['VAE', 'GAN', 'GMM', 'KDE', 'Proposed']
data = {
    'FID':      [153.5658, 129.8420, 166.3631, 179.2621,  45.5560],
    'IS_mean':  [  2.8474,   3.2655,   2.1022,   2.1381,   4.1447],
    'IS_std':   [  0.0811,   0.2122,   0.2720,   0.1842,   0.1659],
    'SSIM':     [  0.58,     0.60,     0.46,     0.35,     0.71  ],
    'PSNR':     [ 15.93,    15.59,    13.60,    10.36,    20.68 ],
}
df = pd.DataFrame(data, index=methods)

# ------------------------------
# Create subplots: 1 row × 4 cols
# ------------------------------
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=False)

# Bar width and positions
x = np.arange(len(methods))
width = 0.6

# FID subplot (lower is better)
ax = axes[0]
ax.bar(x, df['FID'], width, color=plt.cm.Set2.colors)
ax.set_title('FID (↓)')
ax.set_ylabel('Value', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha='right')
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')

# IS subplot with error bars (higher is better)
ax = axes[1]
ax.bar(x, df['IS_mean'], width, yerr=df['IS_std'], capsize=5, color=plt.cm.Set2.colors)
ax.set_title('IS (↑)')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha='right')
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')

# SSIM subplot (higher is better)
ax = axes[2]
ax.bar(x, df['SSIM'], width, color=plt.cm.Set2.colors)
ax.set_title('SSIM (↑)')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha='right')
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')

# PSNR subplot (higher is better)
ax = axes[3]
ax.bar(x, df['PSNR'], width, color=plt.cm.Set2.colors)
ax.set_title('PSNR [dB] (↑)')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha='right')
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')

# Layout and legend
fig.suptitle('Comparison of Image Quality Metrics', fontsize=22, fontweight='bold', y=1.02)
fig.tight_layout()

# Save high-resolution figure
plt.savefig('metrics_comparison_subplots.png', dpi=1200, bbox_inches='tight')
plt.show()
