import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def conf_stats(y_true: pd.Series, y_pred: pd.Series, cut_off=0.5):
    y_pos = (y_pred > cut_off).astype(int)
    
    cm = confusion_matrix(y_true, y_pos).ravel()

    balanced_pred = y_pred.mean()
    
    return cm, balanced_pred

def group_stats(y_true, y_pred,cut_off=0.5):

    cm, balance = conf_stats(y_true, y_pred, cut_off)
    if len(cm) == 4:
        tn, fp, fn, tp = cm
        cm_dict = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    else:
        fp= 0
        tp= 0
        fn=0
        tn = cm[0]
        cm_dict = {'tn': tn, 'fp': 0, 'fn': 0, 'tp': 0}   
    
  
    if any(value == 0 for value in cm_dict.values()):
        for key, value in cm_dict.items(): 
            if value == 0:
                print(str(key) + " is 0, therefore not all metrics can be calculated.")
                
    metrics = {
        **cm_dict,
        
        'Prevalence': (tp + fn) / (tp + fp + tn + fn), # Base rate
        'TPR': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall, Sensitivity
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
        'TNR': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FDR': fp / (tp + fp) if (tp + fp) > 0 else 0,
        'FNR': fn / (tp + fn) if (tp + fn) > 0 else 0,
        'FOR': fn / (tn + fn) if (tn + fn) > 0 else 0,
        'TE': fn / fp if fp > 0 else np.inf,  # Treatment Equality
        'ESP': tp + fp,  # Equal Selection Parity
        'DI': (tp + fp) / (tp + fp + tn + fn),  # Disparate Impact (selection rate)
        'Accuracy': (tp + tn) / (tp + fp + tn + fn),
        'BalancedAcc': ((tp / (tp + fn) if (tp + fn) > 0 else 0) + 
                        (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2,
        'Balance': balance
    }

    return pd.Series(metrics)


def calculate_fairness_metrics(group_a, group_b):
  
    if hasattr(group_a, 'to_dict'):
        group_a = group_a.to_dict()
    if hasattr(group_b, 'to_dict'):
        group_b = group_b.to_dict()
   
    def calculate_ratio(metric_name):
        a_value = group_a.get(metric_name, 0)
        b_value = group_b.get(metric_name, 0)
        
        # Handle division by zero cases
        if a_value == 0 and b_value == 0:
            return 1.0  # Both are equal (perfect fairness)
        elif a_value == 0:
            return float('inf')  # Maximum unfairness in one direction
        else:
            return b_value / a_value       
    
    # Calculate ratio metrics
    ratio_metrics = {
        'PrevR': calculate_ratio('Prevalence'),
        'TPRR': calculate_ratio('TPR'),
        'FNRR': calculate_ratio('FNR'),
        'FPRR': calculate_ratio('FPR'),
        'TNRR': calculate_ratio('TNR'),
        'FORR': calculate_ratio('FOR'),
        'FDRR': calculate_ratio('FDR'),
        'NPVR': calculate_ratio('NPV'),
        'PPVR': calculate_ratio('PPV'),
        'DIR': calculate_ratio('DI')
        }
    
    #create another dictionary with normalized values
    norm_metrics = {f"n_{key}": ratio / (ratio + 1) for key, ratio in ratio_metrics.items()}
        
    
    # Calculate additional fairness metrics
    fairness_metrics = {
        # Difference metrics
        'ESP': group_b.get('ESP', 0) - group_a.get('ESP', 0),
        'TE': group_b.get('TE', 0) - group_a.get('TE', 0),
        'SPD': group_b.get('DI', 0) - group_a.get('DI', 0),
        'Balance': group_b.get('Balance', 0) - group_a.get('Balance', 0),
        
        # Average odds difference
        'AOD': ((group_b.get('TPR', 0) - group_a.get('TPR', 0)) + 
                (group_b.get('FPR', 0) - group_a.get('FPR', 0))) / 2,
        
        # Boolean fairness criteria
        'EOpps': (group_b.get('TPR', 0) - group_a.get('TPR', 0)) == 0,
        'PE': (group_b.get('FPR', 0) - group_a.get('FPR', 0)) == 0,
        'SP': (group_b.get('DI', 0) - group_a.get('DI', 0))  == 0,
        'PP': (group_b.get('PPV', 0) - group_a.get('PPV', 0))  == 0
    }
    
    # Derived boolean criteria
    fairness_metrics['EOdds'] = fairness_metrics['EOpps'] and fairness_metrics['PE']
    fairness_metrics['CUAE'] = (fairness_metrics['PP'] and
                               (group_b.get('NPV', 0) - group_a.get('NPV', 0)) < 0.01)
    
    max_DI = max(group_b.get('DI'),group_a.get('DI')) 
    max_TPR = max(group_b.get('TPR'),group_a.get('TPR')) 
    max_FPR = max(group_b.get('FPR'),group_a.get('FPR')) 
    max_PPV = max(group_b.get('PPV'),group_a.get('PPV')) 
    max_TE = max(group_a.get('TE', 0) ,group_b.get('TE', 0))
    
    norm_fairness_metrics = {
    # Difference metrics
    'ESP': group_b.get('ESP', 0) - group_a.get('ESP', 0),
    'TE': (group_b.get('TE', 0) - group_a.get('TE', 0))/max_TE if max_TE != 0 else None,
    
    'SPD': (group_b.get('DI', 0) - group_a.get('DI', 0)) / max_DI if max_DI != 0 else None,
    'Balance': group_b.get('Balance', 0) - group_a.get('Balance', 0),

    # Average odds difference
    'AOD': 0 if max_TPR== 0 or max_FPR == 0
        else
        ((group_b.get('TPR', 0) - group_a.get('TPR', 0))/max_TPR + 
        (group_b.get('FPR', 0) - group_a.get('FPR', 0))/max_FPR)/ 2,

    # Boolean fairness criteria
    'EOpps': (group_b.get('TPR', 0) - group_a.get('TPR', 0)) / max_TPR == 0 if max_TPR != 0 else None,
    'PE': (group_b.get('FPR', 0) - group_a.get('FPR', 0))/max_FPR == 0 if max_FPR != 0 else None,
    'PP': (group_b.get('PPV', 0) - group_a.get('PPV', 0))/ max_PPV == 0 if max_PPV != 0 else None,
    }

    # Derived boolean criteria
    norm_fairness_metrics['EOdds'] = norm_fairness_metrics['EOpps'] and norm_fairness_metrics['PE']
    norm_fairness_metrics['CUAE'] = (norm_fairness_metrics['PP'] and
                               (group_b.get('NPV', 0) - group_a.get('NPV', 0))/max(group_a.get('NPV', 0) , group_b.get('NPV', 0)) ==0)
    
    # Performance metrics
    performance_metrics = {
        'Acc_Ratio': group_b.get('Accuracy', 0) / group_a.get('Accuracy', 1) 
                     if group_a.get('Accuracy', 0) > 0 else None
    }
    
    # Add BalancedAcc ratio if available
    if 'BalancedAcc' in group_a and 'BalancedAcc' in group_b and group_a.get('BalancedAcc', 0) > 0:
        performance_metrics['BalancedAcc_Ratio'] = (group_b.get('BalancedAcc', 0) / 
                                                   group_a.get('BalancedAcc', 0))

    combined_results = {
    1: ratio_metrics,
    2: norm_metrics,
    3: fairness_metrics,
    4:norm_fairness_metrics,
    5:performance_metrics
}
    return combined_results
