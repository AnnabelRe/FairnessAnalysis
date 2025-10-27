import matplotlib.pyplot as plt
import numpy as np

def bar_chart(df, filename=None, miti="in"):
    
    metrics = ['TE', 'SPD', 'AOD']
    
    m1 = df.iloc[0]
    m2 = df.iloc[1]
    m3 = df.iloc[2]
    m4  = df.iloc[3]

    baseline = [m1[key] for key in metrics]
    miti1 = [m2[key] for key in metrics]
    miti2 = [m3[key] for key in metrics]
    miti3 = [m4[key] for key in metrics]

    baseline = [np.nan if x is None else x for x in baseline]
    miti1 = [np.nan if x is None else x for x in miti1]
    miti2 = [np.nan if x is None else x for x in miti2]
    miti3 = [np.nan if x is None else x for x in miti3]
        
    if miti == "in":

        labels = ['Baseline','FTU','FGBM','FGBM with HPT']
    else: 
        
        labels = ['Baseline','ThresholdOpt','ROC', 'EOdds']

    bar_width = 0.2

    x = np.arange(len(metrics))
    
    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - bar_width/2, baseline, bar_width, label=labels[0], color="#1f77b4")
    bars2 = ax.bar(x + bar_width/2, miti1, bar_width, label=labels[1], color='#ff7f0e')
    bars3 = ax.bar(x - bar_width*1.5, miti2, bar_width, label=labels[2], color='#2ca02c')
    bars4 = ax.bar(x + bar_width*1.5, miti3, bar_width, label=labels[3], color='#d62728')
    
    all_values = baseline+miti1+miti2+miti3
    ax.set_ylim(min(all_values) - 0.1, max(all_values) + 0.1)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.text(len(metrics) - 0.5, 0, ' perfect parity', va='center', ha='left', 
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
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_center()[0], y_pos),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center', va=va,
                        rotation=0,
                        fontsize=10)
    
    add_annotations(bars1)
    add_annotations(bars2)
    add_annotations(bars3)
    add_annotations(bars4)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Fairness Metrics')
    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if filename: 
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()