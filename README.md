# Fairness Analysis in Insurance ML Models
This repository contains the implementation code for my diploma thesis "Towards Fair AI Systems: Identification and Mitigation of Discrimination in the Financial Service Sector" at TU Wien.

## Overview
This project investigates gender and nationality-based discrimination in a real-world insurance machine learning model designed to predict claims likely to "explode" in compensation costs. The research analyzes a Light Gradient Boosting Machine (LightGBM) model used by an Austrian insurance company and explores various fairness mitigation methods. This research was conducted in accordance with EU AI Act requirements for non-discriminatory algorithmic systems, and Austrian legal frameworks for insurance and data protection.  
#### Key Focus Areas  
•	Baseline Model Analysis: LightGBM model evaluation on 450,000 insurance claims  
•	Fairness Assessment: Comprehensive analysis using multiple fairness metrics (Equal Opportunity, Predictive Equality, Equalized Odds, CUAE, Predictive Parity)  
•	In-Processing Mitigation: FairGBM implementation with and without hyperparameter tuning  
•	Fairness Through Unawareness: Testing protected attribute removal  
•	Post-Processing Methods:   
o	Reject Option Classification (AIF360)  
o	Threshold Optimizer (Fairlearn)  
o	Equalized Odds Post-Processing (AIF360)  
•	Performance Evaluation: Analysis with focus on recall due to high class imbalance  

## Key Findings
While mitigation methods successfully improved fairness metrics, improvements came at a severe cost to predictive performance, rendering them impractical for real-world deployment in this case study.

## Usage
Run the Jupyter notebooks in the following order:  
1.	run_models.ipynb - Trains baseline LightGBM and FairGBM models  
2.	mitigation.ipynb - Applies post-processing fairness mitigation methods  
3.	performance_eval.ipynb - Evaluates and compares model performance  
4.	fairness_results.ipynb - Calculates fairness metrics across all groups and models  
5.	miti_comparison.ipynb - Generates visualizations comparing fairness metrics  
Note: All notebooks automatically reference the .py files in the repository. No need to run Python scripts separately.  

## Data Availability
The insurance claims dataset used in this research is confidential and not publicly available due to privacy and regulatory constraints. The code is provided for transparency and reproducibility of the methodology.

## Results & Visualizations
For detailed results, fairness metric visualizations, and additional charts, please contact me via LinkedIn.

## Citation
If you use this code or reference this work, please cite:  
Resch, A. (2025). Towards Fair AI Systems: Identification and Mitigation of Discrimination in the Financial Service Sector [Diploma Thesis, Technische Universität Wien]. reposiTUm. https://doi.org/10.34726/hss.2025.123673

