import optuna
import yaml
import pandas as pd
import fairgbm as fgbm
import numpy as np
from sklearn.metrics import recall_score
from fairgbm.metrics import equalized_opportunity_difference
from sklearn.model_selection import train_test_split
import skops.io as sio

# Load search space from YAML file
with open('fgbm_params.yaml', 'r') as file:
    search_space = yaml.safe_load(file)


X_train = pd.read_pickle("X_train_nos.pkl")
X_test = pd.read_pickle("X_test_nos.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")
S_nat= pd.read_pickle("S_nat.pkl")
S_nat_test= pd.read_pickle("S_nat_test.pkl")
S_gender= pd.read_pickle("S_gender.pkl")
S_gender_test= pd.read_pickle("S_gender_test.pkl")


# Define the objective function for Optuna
def objective(trial):
    # Suggest parameters from the search space
    params = {key: trial.suggest_categorical(key, values) if isinstance(values, list) else
              trial.suggest_float(key, values[0], values[1]) for key, values in search_space.items()}
    
    # Add required static parameters for FairGBM
    params.update({
        "objective": "binary",
        "verbosity": -1,
        "seed": 42
        
    })
  
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    # Train the FairGBM model
    model = fgbm.FairGBMClassifier(**params)    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
    y_pred = model.predict(X_test)
    
    recall = recall_score(y_test, y_pred)
    
    # Calculate the fairness metric (equalized opportunity)
    eo_diff = equalized_opportunity_difference(y_test, y_pred, sensitive_features=S_gender_test)
    
    # Apply fairness constraint: only accept trials where the EO difference is below a threshold (e.g., 0.1)
    fairness_threshold = 0.5
    if eo_diff > fairness_threshold:
        return float('-inf')  # Penalize trials that violate the fairness constraint
    
    return recall  # Maximize recall    

# Create the Optuna study
study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=50)

trials = study.trials_dataframe()
sio.dump(trials,'FGBM_hpt_v1.skops')

# save picture of optimization
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.savefig('FGBM_hpt_v1.jpg', bbox_inches='tight')

# save best trial parameters
sio.dump(study.best_params,'FGBM_v1_best.skops')

# Print the best parameters and score
print("Best trial:")
print("  Params: ", study.best_trial.params)
print("  Recall: ", study.best_value)
