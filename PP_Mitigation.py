import pandas as pd
import numpy as np
import os
import pickle

import skops.io as sio
import seaborn as sns
from matplotlib import pyplot as plt
import lightgbm as lgbm

import json
import pandas as pd

import sys
sys.path.append("../")
import time
from tqdm import tqdm

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric

from reject_option_classification import RejectOptionClassification as ROC
from aif360 import *
from sklearn.metrics import recall_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
#from ipywidgets import interactive, FloatSlider

import warnings
warnings.filterwarnings("ignore")

### Load Data
X_train= pd.read_pickle("X_train_nos.pkl")
X_test = pd.read_pickle("X_test_nos.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

X_test.reset_index(inplace=True, drop=True)
X_train.reset_index(inplace=True, drop=True)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

y_test = y_test.rename("labels")
y_train = y_train.rename("labels")


def run_thresOpt(model, sens_att, constr="true_positive_rate_parity", obj="balanced_accuracy_score"):
    unknown_types = sio.get_untrusted_types(file = model)

    clf=sio.load(model,trusted=unknown_types)

    if sens_att == "gender":
        S_train = pd.read_pickle("S_gender.pkl")
        S_test = pd.read_pickle("S_gender_test.pkl")
    else:   
        S_train = pd.read_pickle("S_nat.pkl")
        S_test = pd.read_pickle("S_nat_test.pkl")

    t0 = time.time()
    threshold_optimizer = ThresholdOptimizer(
    estimator=clf,
    constraints=constr,#equalized_odds demographic_parity, false_positive_rate_parity, false_negative_rate_parity, true_positive_rate_parity, true_negative_rate_parity
    objective=obj,#selection_rate, true_positive_rate, true_negative_rate, accuracy_score, balanced_accuracy_score
    predict_method="predict_proba",
    prefit=False,
    )
    
    threshold_optimizer.fit(X_train, y_train, sensitive_features= S_train,constraint_group=S_train)
    t1 = (time.time()-t0)/60
    t = [f"time_ppTO_{sens_att}_{constr}_{obj}", t1]
    y_pred = threshold_optimizer.predict(X_test, sensitive_features=S_test, random_state=12345)
    threshold_rules_by_group = threshold_optimizer.interpolated_thresholder_.interpolation_dict
    pd.DataFrame(threshold_rules_by_group).to_pickle(f"thresholds_pPTO_{sens_att}_{constr}_{obj}.pkl")

    plot_threshold_optimizer(threshold_optimizer)
    pd.DataFrame(y_pred).to_pickle(f"y_pred_ppTO_{sens_att}_{constr}_{obj}.pkl")
    
    return t

def run_pp_ROC(model, sens_att):

    if sens_att == "gender":
        privileged_groups = [{'gender': 0}]
        unprivileged_groups = [{'gender': 1}]
        S_train = pd.read_pickle("S_gender.pkl")
        S_test = pd.read_pickle("S_gender_test.pkl")  
        S_train = S_train.rename("gender")
        S_test =S_test.rename("gender")      

    else:
        privileged_groups = [{'nationality': 0}]
        unprivileged_groups = [{'nationality': 1}]
        S_train = pd.read_pickle("S_nat.pkl")
        S_test = pd.read_pickle("S_nat_test.pkl")      
        S_train = S_train.rename("nationality")
        S_test =S_test.rename("nationality")  

    t0 = time.time()
    train_data = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=pd.concat([X_train, S_train, y_train], axis=1),
        label_names=[y_train.name],  # True label column
        protected_attribute_names=[S_train.name])

    unknown_types = sio.get_untrusted_types(file = model)

    clf=sio.load(model,trusted=unknown_types)
  
    y_pred_probs = clf.predict_proba(X_train)[:, 1].reshape(-1,1)

    roc = ROC(
        unprivileged_groups=unprivileged_groups, 
        privileged_groups=privileged_groups, 
        low_class_thresh=0.01, 
        high_class_thresh=0.99, 
        num_class_thresh=100, 
        metric_name="Equal opportunity difference",  # Can also use "Statistical Parity Difference"
        metric_ub=0.05,  # Upper bound for fairness constraint
        metric_lb=-0.05  # Lower bound for fairness constraint
    )

    train_data_pred = train_data.copy()
    train_data_pred.scores = y_pred_probs                

    roc.fit(train_data, train_data_pred)

    t1 = (time.time()-t0)/60
    t = [f"time_ppROC_{sens_att}", t1]

    test_data = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=pd.concat([X_test, S_test, y_test], axis=1),
        label_names=[y_test.name],
        protected_attribute_names=[S_test.name]
    )

    test_data_pred = test_data.copy()
    #y_test = test_data_pred.labels
    test_data_pred.scores = clf.predict_proba(X_test)[:,1].reshape(-1,1)

    roc_pred = roc.predict(test_data_pred)

    y_pred_after = roc_pred.labels.ravel()

    pd.DataFrame(y_pred_after).to_pickle(f"y_pred_ppROC_{sens_att}.pkl")

    return t


def run_pp_mitigation(pp, model, sens_att, constr="true_positive_rate_parity", obj="balanced_accuracy_score"):
    if pp == "threshOpt":
        return run_thresOpt(model, sens_att, constr="true_positive_rate_parity", obj="balanced_accuracy_score")
    
    else: return run_pp_ROC(model, sens_att)
