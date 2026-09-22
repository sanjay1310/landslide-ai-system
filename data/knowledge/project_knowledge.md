# Project Knowledge Base

## Platform Summary
The Landslide Intelligence Platform is a hybrid system that combines interpretable scoring, trained machine learning, deep-learning support, graph-based spatial reasoning, forecasting, scenario simulation, API services, and a dashboard interface.

## Main Risk Inputs
The main risk pathway uses rainfall intensity, antecedent rainfall, slope factor, soil wetness, vegetation stress, graph influence, and propagated graph signal.

## Real Data Path
The project includes a Kerala and Uttarakhand regional experiment built from mapped landslide inventory records, NASA POWER daily rainfall, and terrain features such as slope and elevation.

## Graph Model Use
The graph component estimates local spatial influence and propagated network influence between nearby districts. A learned graph model can be blended with heuristic graph exposure when enabled.

## Operational Interpretation
Risk scores should be treated as decision-support signals rather than perfect ground-truth probabilities. Real-data evaluation should prioritize recall, F1, ROC-AUC, and PR-AUC, especially when the event dataset is highly imbalanced.

## Scenario Analysis
The platform can simulate rainfall increase, antecedent rainfall stress, soil wetness changes, NDVI shifts, and neighbor-risk changes to study how district risk patterns move under different environmental assumptions.
