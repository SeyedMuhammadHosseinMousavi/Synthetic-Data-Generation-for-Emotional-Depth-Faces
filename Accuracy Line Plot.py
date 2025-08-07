import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Configure fonts and style
# ------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 18,
    'font.weight': 'bold',
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 18,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# ------------------------------
# Accuracy data only
# ------------------------------
accuracy = {
    'RF (Synthetic)':    {'VAE':81.67, 'GAN':85.42, 'GMM':65.42, 'KDE':60.42, 'Proposed':91.67},
    'RF (Syn + Org)':    {'VAE':90.21, 'GAN':90.42, 'GMM':77.08, 'KDE':77.50, 'Proposed':92.50},
    'DT (Synthetic)':    {'VAE':52.92, 'GAN':57.50, 'GMM':51.50, 'KDE':46.25, 'Proposed':58.92},
    'DT (Syn + Org)':    {'VAE':67.50, 'GAN':66.42, 'GMM':65.21, 'KDE':60.00, 'Proposed':68.92},
    'XGB (Synthetic)':   {'VAE':86.25, 'GAN':88.75, 'GMM':69.58, 'KDE':61.67, 'Proposed':94.58},
    'XGB (Syn + Org)':   {'VAE':93.96, 'GAN':94.17, 'GMM':81.25, 'KDE':76.67, 'Proposed':96.46},
}

# Create DataFrame (rows = classifiers, columns = methods)
df_acc = pd.DataFrame(accuracy).T

# ------------------------------
# Plot a wide, thick, bold line plot
# ------------------------------
fig, ax = plt.subplots(figsize=(24, 6))

df_acc.plot(
    kind='line',
    marker='o',
    linewidth=3,      # thicker lines
    markersize=8,
    ax=ax
)

# Titles and labels in bold
ax.set_title('Classifier Accuracy Comparison', fontweight='bold', pad=12)
ax.set_ylabel('Accuracy (%)', fontweight='bold')

# Correct tick positions and labels
ax.set_xticks(range(len(df_acc.index)))
ax.set_xticklabels(df_acc.index, rotation=30, ha='right')

# Bold tick labels
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')

# Bold legend text
legend = ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1.02, 1))
legend.get_title().set_fontweight('bold')
for text in legend.get_texts():
    text.set_fontweight('bold')

# ------------------------------
# Save at high resolution
# ------------------------------
plt.tight_layout()
plt.savefig('accuracy_comparison_lineplot.png', dpi=1200, bbox_inches='tight')
plt.show()
