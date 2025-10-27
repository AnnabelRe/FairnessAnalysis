from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np
import seaborn as sns

def radar_chart(df, ideal_value=1.0, filename=None):
    """
    Create a radar chart inlcuding fairness metrics of all mitigation methods
    """
    categories = list(df.columns)
    N = len(categories)
   
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
   
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Set 0 degrees to top (12 o'clock)
    ax.set_theta_offset(np.pi / 2)  # Offset by 90 degrees
    ax.set_theta_direction(1)  # Clockwise
    
    # Draw one axis per variable + add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    ideal_values = [ideal_value] * N
    ideal_values += ideal_values[:1]
    ax.plot(angles, ideal_values, 'r--', linewidth=2, label='Ideal Ratio (1.0)')
    
    #colors = sns.color_palette("Greens")
    
    for i, model in enumerate(df.index):
        values = df.loc[model].values.tolist()
        values += values[:1]  
        ax.plot(angles, values, linewidth=2, label=model)
        #ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Fair Models: Fairness Ratios Comparison', size=15)
    
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
