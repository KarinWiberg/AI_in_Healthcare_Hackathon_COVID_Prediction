# AI_in_Healthcare_Hackathon_COVID_Prediction
Leveraging Clinical Data to Advance on the Clinical Management of COVID-19 (Hospital Data Set): use of an open data sets containing anonymized EHR data from thousands of COVID-19 patients to advance our knowledge, prediction, treatment and overall understanding of COVID-19.

## Solution
Help reduce covid-related death and to help the management of covid patients in the hospital by prediction for the patient risk from cheap, early and easy accessible metrics.

## What it does
Receives input in the form of clinical statistics or an x-ray image and predicts whether the patient's condition will worsen or get better.

Predicts the patient risks in two levels
- Recovery
- Death
Also, it delivers the features of importance for each feature.

## How we built it
Parts in Jupyter Notebook and parts in python. The data was cleaned and features were added to bolster the model's accuracy. The death/recovery models were built in sklearn and had their hyperparameters optimized by the TPOT genetic algorithm. The image classification model was build in keras using a CNN.

## Accomplishments that we're proud of
We created a predictive pipeline to determine the outcome of a person's current covid condition. Our death prediction model achieved 95% accuracy and 95% recall while our CNN achieved 100% validation accuracy and recall on our sample size! We are proud of the collaboration, which included people from all over the world, where people came together to solve an important topic.

# Folder Structure
## scripts
Main scripts for cleaning, exploring and analysis.
### Cleaning & Feature Selection
1. Batch 1: feature_selection_cleaning_v2.ipynb
2. Batch 2: feature_selection_cleaning-batch-2_v2.ipynb
3. Batch 3: feature_selection_cleaning-batch-3_v2.ipynb
4. Combine the 3 clean batches: combine_clean_data_v2.ipynb

### Prediction Models
Death Prediction: best_death_pipeline_final.py
Death and Recovery Prediction: explorative-analysis-clean-combined-data-v2-final.ipynb  
Death Prediction X-Ray Images: xray_classification.py

## data
The data is located in the data folder and provided by HM Hospitales.

## utils
Internal libraries.

## visualizations
Main feature importance visualizations for:  
Death prediction: death_random_forest_feature_importance_final.png  
Recovery prediction: recovery_random_forest_feature_importance_final.png
