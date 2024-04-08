import streamlit as st

def display(tab):
    tab.subheader("PyCaret for Polymer Property Prediction")

    tab.write("""
    In this tutorial, we focus on utilizing the PyCaret library to predict polymer properties based on SMILES strings, which are textual representations of chemical structures. We will preprocess these SMILES strings to generate molecular fingerprints, transforming them into a format suitable for machine learning models. You can find the dataset used in this tutorial at [this link](https://khazana.gatech.edu/dataset/).

    PyCaret is an AutoML library that simplifies many of the steps involved in the machine learning pipeline, including data preprocessing, model selection, and hyperparameter tuning. This makes it an ideal choice for quickly developing and comparing different models to predict the properties of polymers.
    """)

    tab.subheader("Step 1: Install Necessary Libraries")
    tab.code("""
%pip install pycaret
%pip install psmiles
%pip install pandas
%pip install numpy
%pip install matplotlib
""")

    tab.subheader("Step 2: Importing Libraries and Loading Data")
    tab.code("""
from pycaret.regression import *
import pandas as pd
import numpy as np
from psmiles import PolymerSmiles as PS

# Load and preprocess data
data_path = 'your_data_path/export.csv'
data_df = pd.read_csv(data_path)
data_df = data_df[data_df.property == 'Egc']
""")

    tab.subheader("Step 3: Data Preprocessing and Feature Engineering")
    tab.code("""
# Generate molecular fingerprints from SMILES strings
smile_strings = data_df['smiles'].values
fingerprints = [PS(smile).fingerprint() for smile in smile_strings]
data_df['fingerprints'] = fingerprints

# Preparing the final dataset
features_df = pd.DataFrame(data_df['fingerprints'].tolist(), index=data_df.index)
final_df = pd.concat([features_df, data_df['Egc']], axis=1)
""")

    tab.subheader("Step 4: Setting up the Environment in PyCaret")
    tab.code("""
from pycaret.regression import setup, compare_models, tune_model, predict_model

# Initialize the PyCaret environment
exp_reg = setup(data=final_df, target='Egc', silent=True, session_id=123)
""")

    tab.subheader("Step 5: Comparing and Selecting Models")
    tab.code("""
best_model = compare_models()
""")

    tab.subheader("Step 6: Model Tuning")
    tab.code("""
tuned_model = tune_model(best_model)
""")

    tab.subheader("Step 7: Model Evaluation")
    tab.code("""
predictions = predict_model(tuned_model)
""")

    tab.subheader("Step 8: Visualization of Predictions")
    tab.code("""
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(predictions['Label'], predictions['Egc'], edgecolors='k')
plt.plot([predictions['Egc'].min(), predictions['Egc'].max()], [predictions['Egc'].min(), predictions['Egc'].max()], 'r--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Prediction Accuracy')
plt.show()
""")

    tab.subheader("Conclusion")
    tab.write("""
    Through this tutorial, we have explored the process of using PyCaret to predict polymer properties from SMILES strings. From data preprocessing and feature engineering to model training and evaluation, PyCaret offers a comprehensive and accessible AutoML solution for rapid model development.
    """)