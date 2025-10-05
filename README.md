NASA EXOHUNTERS
========================

Overview
--------
This project uses NASA Kepler Space Telescope data to classify celestial bodies
as Confirmed Exoplanets, Candidates, or False Positives using a Random Forest 
Machine Learning model.

Key Features
-------------
• Data cleaning and preprocessing  
• Class balancing with SMOTE  
• Model training using RandomForestClassifier  
• Visualizations: Feature Importance and Confusion Matrix  
• Interactive prediction for new exoplanet data  
• CSV exports for prediction results  

Dataset
--------
Kepler Exoplanet data sourced from NASA’s public archive:
https://raw.githubusercontent.com/Siddharths99/NASA-Exoplanet-Detection/refs/heads/main/data/kepler.csv

Model Summary
--------------
Algorithm : Random Forest Classifier  
Accuracy  : ~77%  
Key Features : koi_prad, koi_teq, koi_period  

Outputs
--------
• exo_predictions_with_names.csv  → Full model predictions  
• top_exo_candidates_with_names.csv  → Top potential exoplanets  

Run Instructions
-----------------
Install dependencies:
    pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib ipywidgets

Run the notebook in google collab or run the src in vs code

Example Prediction
-------------------
Predicted Class: CONFIRMED  
Prediction Probability: 0.89
