import matplotlib.pyplot as plt
import numpy as np

def bar_chart(dictionary, norm, filename):
    labels = ['TE', 'SPD', 'AOD']
    values = [dictionary[key] for key in labels]
    norm_values = [norm[key] for key in labels]
    
    values = [0 if x is None or (isinstance(x, float) and x != x) else x for x in values]
    norm_values = [0 if x is None or (isinstance(x, float) and x != x) else x for x in norm_values]
    

    bar_width = 0.35
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars1 = ax.bar(x - bar_width/2, values, bar_width, label='Original', color='cornflowerblue')
    bars2 = ax.bar(x + bar_width/2, norm_values, bar_width, label='Normalized', color='springgreen')
    

    all_values = values + norm_values
    ax.set_ylim(min(all_values) - 0.1, max(all_values) + 0.1)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.text(len(labels) - 0.5, 0, ' perfect parity', va='center', ha='left', 
            fontsize=10, fontweight='bold', backgroundcolor='white', alpha=0.8)
    
    # Add value annotations on top of bars
    def add_annotations(bars):
        for bar in bars:
            height = bar.get_height()
            # Position annotation based on whether value is positive or negative
            if height < 0:
                va = 'top'
                y_pos = height - 0.002
            else:
                va = 'bottom'
                y_pos = height + 0.002
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_center()[0], y_pos),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center', va=va,
                        rotation=0,
                        fontsize=10)
    
    add_annotations(bars1)
    add_annotations(bars2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Fairness Metrics')
    ax.legend()
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if filename: 
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()