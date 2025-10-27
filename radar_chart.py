import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars):
    """
    Create a radar chart with `num_vars` axes.
    """
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            
        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)
            
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
                
        def _close_line(self, line):
            x, y = line.get_data()
            
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)
                
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
    
    register_projection(RadarAxes)
    return theta


def plot_fairness_radar(ratios, normalized_ratios, filename=None):
    """
    Create a radar chart comparing original and normalized fairness metrics.
    
    Args:
        ratios: Dictionary of original ratio values
        normalized_ratios: Dictionary of normalized ratio values
        filename: If provided, save the figure to this file
    """
   
    metrics = sorted(ratios.keys())
    N = len(metrics)
    
 
    theta = radar_factory(N)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    
    ratio_values = [ratios[metric] for metric in metrics]
    normalized_values = []
    for metric in metrics:
        normalized_key = f"n_{metric}"
        if normalized_key in normalized_ratios:
            normalized_values.append(normalized_ratios[normalized_key])
        else:
            normalized_values.append(None)
    
    # Plot original ratios
    ax.plot(theta, ratio_values, 'o-', linewidth=2, color='cornflowerblue',label='Original Ratios')
    ax.fill(theta, ratio_values, alpha=0.1,color='cornflowerblue')
    
    # Plot normalized ratios
    ax.plot(theta, normalized_values, 'o-', linewidth=2,color='springgreen',label='Normalized Ratios')
    ax.fill(theta, normalized_values, alpha=0.1,color='springgreen')
    
    ax.spines['polar'].set_visible(False)
    
    # Add reference line for parity (value = 1 for original, 0.5 for normalized)
    ax.plot(theta, [1] * N, '--', linewidth=1, color='gray', label='Parity (Original)')
    ax.plot(theta, [0.5] * N, '--', linewidth=1, color='red', label='Parity (Normalized)')

    ax.set_varlabels(metrics)
    ax.set_title('Fairness Metrics Comparison', size=15, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    max_original = max(ratio_values)
    plt.ylim(0, max(2, max_original * 1.2))  
    
    ax.grid(True)
    
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

