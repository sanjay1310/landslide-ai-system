# Project Report

## on

# Landslide Intelligence Platform:
# Hybrid AI-Based Monitoring, Forecasting, and Spatial Advisory System

Submitted for partial fulfilment of the requirements for the award of the degree of  
**BACHELOR OF ENGINEERING**  
in  
**COMPUTER SCIENCE AND ENGINEERING (AI \& ML)**

By  
`[Student Name 1] ([Roll Number 1])`  
`[Student Name 2] ([Roll Number 2])`  
`[Student Name 3] ([Roll Number 3])`

Under the guidance of  
`[Guide Name]`  
`[Designation]`

**Department of Computer Science and Engineering**  
`[College Name]`  
`[Autonomous / Affiliation details]`  
`[College Address]`  
**Academic Year: 2025-26**

---

# CERTIFICATE

This is to certify that the project work entitled **"Landslide Intelligence Platform: Hybrid AI-Based Monitoring, Forecasting, and Spatial Advisory System"** is a bonafide work carried out by `[Student Name 1]`, `[Student Name 2]`, and `[Student Name 3]` in fulfilment of the requirements for the award of the degree of Bachelor of Engineering in Computer Science and Engineering during the Academic Year 2025-26 under our guidance and supervision.

The results embodied in this report have not been submitted to any other university or institute for the award of any degree or diploma to the best of our knowledge and belief.

**Internal Guide**  
`[Guide Name]`  
`[Designation]`

**Head of the Department**  
`[HOD Name]`  
`[Designation]`

---

# DECLARATION

This is to certify that the work reported in the present project entitled **"Landslide Intelligence Platform: Hybrid AI-Based Monitoring, Forecasting, and Spatial Advisory System"** is a record of bonafide work done by us in the Department of Computer Science and Engineering during the Academic Year 2025-26.

The report is based on the project work done entirely by us and not copied from any other source. The results embodied in this project report have not been submitted to any other University or Institute for the award of any degree or diploma.

`[Student Name 1]`  
`[Student Name 2]`  
`[Student Name 3]`

---

# ACKNOWLEDGEMENT

We would like to express our sincere gratitude to our project guide `[Guide Name]`, `[Designation]`, for valuable suggestions, constant support, and guidance throughout the course of this project.

We are also thankful to the Head of the Department, faculty members, and the management of `[College Name]` for providing the required academic environment, infrastructure, and encouragement to complete this project successfully. We extend our gratitude to the laboratory staff and all those who directly or indirectly helped us during the development, testing, and documentation of this work.

We also acknowledge the contribution of public data and technical resources used in the project, including regional terrain records, NASA POWER rainfall services, and landslide inventory data used for research-oriented experimentation.

---

# ABSTRACT

This project presents a software-only **Landslide Intelligence Platform** for district-level landslide monitoring, forecasting, and advisory generation. The system is designed to support regional risk assessment using rainfall, terrain, vegetation-related stress, and spatial neighborhood influence. Unlike hardware-dependent landslide warning systems, this platform is deployable as a software framework and can operate through command-line execution, API-based backend services, and a web dashboard.

The project combines multiple analytical components. A baseline interpretable risk model estimates landslide susceptibility using rainfall intensity, antecedent rainfall, slope, soil wetness, vegetation stress, and graph-based spillover features. In addition to this, trained machine learning models, a graph neural network path for spatially structured learning, and a deep-learning-based risk model were integrated to improve predictive capability. The platform also includes forecasting support, scenario simulation, graph-aware hotspot interpretation, and advisory generation for operational use.

To strengthen the research value of the project, the system was extended from a small prototype dataset to a larger district-date temporal dataset. Further, a real-data experiment was conducted for Kerala and Uttarakhand by combining mapped landslide inventory records, NASA POWER daily rainfall for district centroids, and terrain features such as slope and elevation. This resulted in an inventory-labeled regional dataset with 81,405 district-date rows, enabling more realistic evaluation under class imbalance. The platform therefore serves both as a deployable academic project and as a foundation for future publication-oriented landslide risk research.

---

# TABLE OF CONTENTS

1. Introduction  
1.1 Problem Statement  
1.2 Objectives  
1.3 Motivation  
1.4 Scope  
1.5 Software and Hardware Requirements  

2. Literature Survey  
2.1 Major Areas  
2.2 Techniques and Algorithms  
2.3 Applications  

3. System Design  
3.1 System Architecture  
3.2 Module Description  
3.3 Project Plan  

4. Implementation  
4.1 Environment Setup  
4.2 Implementation of Each Module  
4.3 Integration and Deployment  

5. Testing and Results  
5.1 Software Testing  
5.2 Test Cases  
5.3 Results  

6. Conclusion and Future Enhancements  
6.1 Conclusion  
6.2 Future Enhancements  

References  
Appendix

---

# LIST OF FIGURES

Fig 3.1 System Architecture  
Fig 3.2 End-to-End Workflow  
Fig 4.1 Dashboard Overview  
Fig 4.2 API and Service Flow  
Fig 5.1 Risk Hotspot Visualization  
Fig 5.2 Real-Data Evaluation Summary

---

# LIST OF TABLES

Table 3.1 Project Schedule  
Table 5.1 Snapshot Model Comparison  
Table 5.2 Real Kerala-Uttarakhand Evaluation  
Table 5.3 Confusion Matrix Summary

---

# CHAPTER 1: INTRODUCTION

## 1.1 Problem Statement

Landslides are one of the most dangerous natural hazards in hilly and high-rainfall regions. Their occurrence is strongly influenced by factors such as heavy rainfall, antecedent soil saturation, terrain slope, vegetation condition, and the regional influence of neighboring unstable zones. Traditional warning systems often depend on manual observation, isolated threshold methods, or expensive sensing infrastructure, which limits scalability and real-world accessibility.

The problem addressed in this project is the development of a software-based, district-level landslide risk platform that can ingest environmental data, compute present and future risk, identify spatial hotspots, and produce human-readable advisories for decision support.

## 1.2 Objectives

- To build a software-only landslide monitoring and advisory system.
- To integrate rainfall, terrain, vegetation, and graph-based spatial features into one risk framework.
- To support both interpretable and trained prediction paths.
- To provide forecasting, scenario simulation, and regional comparison features.
- To expose the system through CLI, API, and dashboard interfaces.
- To prepare the architecture for real-data research using inventory labels and NASA POWER rainfall.

## 1.3 Motivation

Landslide-prone states such as Kerala and Uttarakhand experience recurring rainfall-triggered slope failures that affect transport, settlements, and human life. Existing academic projects often stop at static classification or do not offer a deployable operational workflow. This motivated the development of a platform that is not only predictive but also modular, explainable, and presentation-ready.

The project was also motivated by the need to convert a classroom prototype into a publication-oriented research framework. For that reason, the platform was later extended to support larger temporal datasets, trained models, deep learning, graph-based reasoning, and real regional rainfall-event experiments.

## 1.4 Scope

The scope of this project includes district-level landslide risk estimation, rainfall-based forecast support, graph-assisted hotspot reasoning, and advisory generation. The platform currently supports India-focused regional analysis and has been extended specifically for Kerala and Uttarakhand real-data experiments.

The present scope does not yet include fully automated live production deployment, field-calibrated warning thresholds, or dense satellite raster modeling. However, the architecture is designed to support these upgrades in future work.

## 1.5 Software and Hardware Requirements

### Software Requirements
- Python 3.x
- Streamlit
- FastAPI
- scikit-learn
- pandas
- NumPy
- PyTorch (for deep model support)
- VS Code / terminal environment

### Hardware Requirements
- Standard laptop or desktop system
- Minimum 8 GB RAM recommended
- Internet connectivity for NASA POWER rainfall download

---

# CHAPTER 2: LITERATURE SURVEY

## 2.1 Major Areas

The literature around landslide prediction can be broadly grouped into:
- landslide inventory mapping,
- rainfall-threshold-based warning systems,
- susceptibility modeling using machine learning,
- remote-sensing-based terrain and vegetation analysis,
- and deep learning or graph learning approaches.

Inventory quality is a major factor in reliable landslide modeling. Studies have shown that poor inventory labeling can significantly weaken the downstream performance of susceptibility models. Rainfall-triggered landslide studies also emphasize the importance of using both immediate rainfall and antecedent moisture conditions.

## 2.2 Techniques and Algorithms

Traditional methods include rainfall thresholds, logistic regression, and rule-based hazard estimation. More recent methods include random forests, boosting methods, neural networks, and graph neural networks. Deep-learning-based susceptibility models can capture non-linear interactions, but they require careful calibration, especially in imbalanced event datasets.

In this project, a hybrid approach was adopted:
- rule-based baseline risk scoring,
- trained random-forest-based risk scoring,
- deep feed-forward neural risk scoring,
- graph-based spillover modeling,
- and temporal district-date evaluation using inventory-based labels.

## 2.3 Applications

Applications of landslide intelligence systems include:
- district-level monitoring,
- hotspot prioritization,
- road and infrastructure safety analysis,
- scenario planning for rainfall escalation,
- and advisory generation for disaster management workflows.

The current project is positioned as a decision-support platform rather than a final operational warning authority.

---

# CHAPTER 3: SYSTEM DESIGN

## 3.1 System Architecture

The system follows four major layers:
- Ingestion Layer
- Feature Fusion Layer
- Inference Layer
- Presentation Layer

Raw data sources include district rainfall, terrain attributes, and landslide event inventory records. These are fused into model-ready district or district-date datasets. Agent-based modules then compute vegetation stress, graph spillover, current risk, forecast risk, and advisories. Results are presented through dashboard and API interfaces.

## 3.2 Module Description

### 3.2.1 Ingestion Module
- prepares raw district records,
- supports NASA POWER rainfall ingestion,
- supports inventory mapping for Kerala and Uttarakhand.

### 3.2.2 Feature Engineering Module
- computes rainfall windows,
- antecedent rainfall,
- soil wetness proxy,
- vegetation vulnerability,
- terrain-rainfall interactions,
- and graph-influence features.

### 3.2.3 Prediction Module
- baseline interpretable model,
- trained tabular risk model,
- deep-risk model for main prediction.

### 3.2.4 Graph Module
- models neighborhood spillover,
- supports graph hotspot interpretation,
- retains extensibility for stronger GNN-style work.

### 3.2.5 Forecast Module
- supports rainfall prediction artifacts,
- enables forecast-informed risk reporting.

### 3.2.6 Advisory Module
- converts scores into human-readable advisories,
- identifies primary and secondary drivers.

### 3.2.7 Dashboard and API Module
- dashboard for visual operational use,
- API for backend/service access,
- exports, tables, scenario view, and analytics.

## 3.3 Project Plan

| Phase | Activity | Outcome |
|---|---|---|
| Phase 1 | Base platform design | Modular landslide pipeline |
| Phase 2 | Baseline risk model | Interpretable scoring system |
| Phase 3 | Trained and deep models | Improved predictive paths |
| Phase 4 | Dashboard and API | Deployable interfaces |
| Phase 5 | Research extension | Temporal evaluation pipeline |
| Phase 6 | Real-data integration | Kerala-Uttarakhand inventory + NASA POWER experiment |

---

# CHAPTER 4: IMPLEMENTATION

## 4.1 Environment Setup

The project was implemented in Python using a modular package structure. The environment includes model training utilities, data ingestion scripts, dashboard code, API services, and testing scripts. Deep-learning support was added using PyTorch in the local virtual environment.

## 4.2 Implementation of Each Module

### 4.2.1 Data Preparation

Initial district-level data was stored in project CSV files. Later, a real-data workflow was added using:
- raw landslide inventory CSV,
- district mapping,
- NASA POWER daily rainfall download,
- terrain merge,
- district-date risk dataset construction.

### 4.2.2 Risk Assessment Logic

The first version of the project used a rule-based risk model. It combined weighted signals from rainfall, antecedent rainfall, slope, soil wetness, vegetation stress, and graph influence. This baseline was retained as a fallback for interpretability.

### 4.2.3 Trained Risk Model

A supervised random forest risk model was introduced on the snapshot dataset. This model improved the validation accuracy and F1 score over the pure rule-based baseline.

### 4.2.4 Deep Risk Model

A feed-forward neural network was added as the primary deep-learning path. It uses seven core landslide features and was trained successfully on the project’s snapshot dataset. The model output was later softened and blended to avoid unrealistic overconfident scores.

### 4.2.5 Real Kerala-Uttarakhand Pipeline

The real-data pipeline:
- maps locality-style inventory records to project districts,
- downloads NASA POWER rainfall by district centroid,
- saves rainfall locally,
- merges rainfall, terrain, and inventory labels,
- and builds an 81,405-row district-date training dataset.

### 4.2.6 Dashboard and API

The dashboard was built using Streamlit and includes:
- operational summary cards,
- maps,
- state and district profiles,
- alerts,
- scenario analysis,
- forecasting view,
- and exports.

The API was built using FastAPI and supports:
- assessment,
- scenario simulation,
- forecasting,
- analytics,
- and configuration status.

## 4.3 Integration and Deployment

The final project can run in three modes:
- CLI mode,
- API mode,
- dashboard mode.

This allows the project to function as both an academic demonstration platform and a base for future deployment-oriented work.

---

# CHAPTER 5: TESTING AND RESULTS

## 5.1 Software Testing

The project was tested at multiple levels:
- module-level testing,
- dashboard runtime testing,
- API startup and route testing,
- training and evaluation script testing,
- real-data pipeline validation.

Different issues related to Streamlit compatibility, caching, export paths, and dataset merging were identified and fixed during integration.

## 5.2 Test Cases

| Test Case | Description | Status |
|---|---|---|
| Dashboard load | Dashboard renders main sections | Passed |
| API startup | FastAPI server initializes successfully | Passed |
| Snapshot training | Trained model artifacts generated | Passed |
| Deep risk training | Deep model trained successfully in virtual environment | Passed |
| Real-data inventory mapping | Raw event CSV mapped to districts | Passed |
| NASA POWER ingestion | Rainfall downloaded for Kerala/Uttarakhand | Passed |
| Real temporal dataset build | Inventory-labeled dataset generated | Passed |
| Real evaluation | Temporal model metrics generated | Passed |

## 5.3 Results

### 5.3.1 Snapshot Results

On the 59-district snapshot dataset:
- Rule-based baseline:
  - Accuracy: `0.831`
  - F1: `0.737`
  - ROC-AUC: `0.941`
- Trained random forest:
  - Validation Accuracy: `0.933`
  - Validation F1: `0.923`
- Deep risk model:
  - Validation Accuracy: `1.000`
  - Validation F1: `1.000`

These snapshot results showed that the trained and deep paths were integrated correctly, but the dataset was too small to claim strong generalization from these metrics alone.

### 5.3.2 Real Kerala-Uttarakhand Results

Using the strengthened real-data regional experiment:
- Dataset rows: `81,405`
- Positive windows after merging NASA GLC events: `893`
- Label source: `merged inventory`
- Rainfall source: `NASA POWER`

Best full-dataset traditional model:
- `logistic_regression_balanced`
- Accuracy: `0.9808`
- Precision: `0.0914`
- Recall: `0.1184`
- F1: `0.1032`
- ROC-AUC: `0.8674`
- PR-AUC: `0.0662`

The graph pipeline was also strengthened before the final comparison. Edge weights now combine geographic distance with rainfall similarity, slope similarity, temperature/vegetation similarity, and shared regional-terrain membership. The graph-ready feature frame was extended with terrain ruggedness, drainage density proxies, river-distance proxies, road-density proxies, geology and land-cover risk proxies, and rolling historical landslide frequency. Operational alerts were upgraded to use a calibrated risk score and confidence-aware action bands rather than only raw low/moderate/high labels.

Comparison with updated candidates:
- Logistic regression (balanced) remained the best practical full-dataset model on the merged real dataset.
- The calibrated logistic variant achieved similar F1, better precision, and a much better Brier score, but slightly lower recall.
- Gradient boosting remained competitive on ROC-AUC, but its thresholded recall collapsed on the merged-label benchmark.
- A reduced temporal GNN benchmark on the same merged-label dataset achieved `0.2074` precision, `1.0000` recall, `0.3435` F1, and `0.1456` PR-AUC on a graph-tractable real-data slice, making the GNN path the main differentiator of the project.
- A first spatio-temporal GNN benchmark on the merged district-date graph-sequence dataset achieved `0.7943` ROC-AUC, `0.0422` PR-AUC, and `0.9836` recall across `20,304` node-date validation predictions, showing that graph-sequence modeling improves ranking quality over the reduced static GNN even though thresholded precision and F1 remain weak in the first run.

### 5.3.3 Fair Same-Scope Benchmark

To solve the earlier comparison weakness, a fair benchmark was generated on the same district-date graph-sequence holdout for three models: the traditional runtime model, the static graph GNN, and the spatio-temporal GNN. Thresholds were tuned on a separate calibration slice from the same benchmark before final holdout evaluation.

Fair same-scope holdout results:
- Traditional runtime model:
  - Precision: `0.0182`
  - Recall: `0.8824`
  - F1: `0.0356`
  - ROC-AUC: `0.8484`
  - PR-AUC: `0.0435`
- Static graph GNN:
  - Precision: `0.0067`
  - Recall: `1.0000`
  - F1: `0.0133`
  - ROC-AUC: `0.5000`
  - PR-AUC: `0.0067`
- Spatio-temporal GNN (advanced GRU):
  - Precision: `0.0422`
  - Recall: `0.8235`
  - F1: `0.0803`
  - ROC-AUC: `0.9099`
  - PR-AUC: `0.1324`

This fair benchmark is now the strongest evidence in the project, because it removes the earlier scope mismatch and also shows a genuine advantage for the advanced model. On the shared graph-sequence holdout, the upgraded spatio-temporal GNN clearly beats the traditional runtime model and the static GNN on ROC-AUC, PR-AUC, and F1. The traditional runtime model only avoids an all-zero operating point by using a very low threshold, while the static graph GNN becomes over-sensitive and predicts nearly every row as positive. This makes the ST-GNN the best head-to-head model rather than only a promising future direction.

### 5.3.4 Interpretation

The real-data results are much more difficult than the snapshot results because the dataset is highly imbalanced and based on mapped inventory labels. This is scientifically more meaningful than proxy-label perfect scores. After merging NASA Global Landslide Catalog events into the inventory, the positive rate rose from `0.69%` to about `1.10%`, and the best logistic model improved its ranking quality to `0.8674` ROC-AUC and `0.0662` PR-AUC on the full merged temporal benchmark. More importantly, the fair same-scope benchmark now shows that graph-based learning is not only a conceptual differentiator. On the common district-date graph-sequence holdout, the upgraded spatio-temporal GNN became the strongest overall model and substantially improved ranking quality while preserving a more useful operating trade-off than the other two approaches. This makes the project significantly stronger for submission and much more defensible for publication.

---

# CHAPTER 6: CONCLUSION AND FUTURE ENHANCEMENTS

## 6.1 Conclusion

This project successfully developed a **Landslide Intelligence Platform** for district-level monitoring, forecasting, and advisory generation. The system evolved from a rule-based prototype into a hybrid AI framework containing trained and deep-learning-based risk paths, graph-informed reasoning, a deployable dashboard, an API backend, and a research evaluation pipeline.

The project’s major achievement is the successful integration of real regional experimentation using mapped landslide inventory data and NASA POWER daily rainfall for Kerala and Uttarakhand. This significantly improved the scientific value of the work by moving beyond only proxy labels and synthetic evaluation.

## 6.2 Future Enhancements

- Improve event-to-district mapping using more reliable official district geospatial matching.
- Add NDVI and richer terrain/environmental covariates.
- Improve the spatio-temporal GNN with longer sequence windows, stronger rare-event tuning, and better calibrated operating thresholds.
- Improve graph modeling with harmonized real geology, land-cover, river, and road layers instead of proxy-derived placeholders.
- Add PR-curve and calibration-based operational threshold tuning.
- Extend the real-data pipeline to more Indian states.
- Add scheduled ingestion and near-real-time alerting support with a production scheduler.

---

# REFERENCES

1. Froude, M. J., and Petley, D. N. Global fatal landslide occurrence from 2004 to 2016. *Natural Hazards and Earth System Sciences*, 2018.  
2. Guzzetti, F. et al. Landslide inventory maps: New tools for an old problem. *Earth-Science Reviews*, 2012.  
3. Hao, L. et al. Constructing a complete landslide inventory dataset for the 2018 monsoon disaster in Kerala, India. *Earth System Science Data*, 2020.  
4. Azarafza, M. et al. Deep learning-based landslide susceptibility mapping. *Scientific Reports*, 2021.  
5. Youssef, K. et al. Landslide susceptibility modeling by interpretable neural network. *Communications Earth \& Environment*, 2023.  
6. Gonzalez, F. C. G. et al. A systematic review on rainfall thresholds for landslides occurrence. *Heliyon*, 2023.  
7. NASA POWER Project Documentation. Daily meteorological data services and API reference.

---

# APPENDIX

## Appendix A: Important Files in the Project

- `src/landslide_ai/dashboard/app.py`
- `src/landslide_ai/api/app.py`
- `src/landslide_ai/agents/prediction_agent.py`
- `src/landslide_ai/models/baseline.py`
- `src/landslide_ai/models/trained_risk.py`
- `src/landslide_ai/models/deep_risk.py`
- `src/landslide_ai/research/large_scale.py`
- `scripts/build_kerala_uttarakhand_real_dataset.py`

## Appendix B: Real-Data Outputs

- `data/inventory/kerala_uttarakhand_inventory_mapped.csv`
- `data/regional/nasa_power_daily_kerala_uttarakhand.csv`
- `data/regional/kerala_uttarakhand_risk_timeseries.csv`
- `artifacts/kerala_uttarakhand_real_evaluation.json`
- `docs/KERALA_UTTARAKHAND_REAL_EVALUATION.md`
