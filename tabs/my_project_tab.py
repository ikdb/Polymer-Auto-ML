# tab2.py
import streamlit as st
import pandas as pd
from tabs.my_project_tabs import auto_gluon_tab,auto_sklearn_tab,auto_keras_tab,pycaret_tab,ludwig_tab

def display(tab):

    tab.header("Automated Machine Learning for Predicting Polymer Properties")
    tab.write("""
    Project Overview:
    Embarking on an exciting venture, I'm here to introduce my initiative: an exploration into the realm of AutoML libraries. This project is an adventure in assessing tools like Autogluon, Ludwig, AutoML, autosklearn, autokeras, and pycaret, aiming to revolutionize efficiency in predictive model generation within polymer impact behavior studies.
    """)
    data_for_table = {
        'Rank': [1, 2, 3, 4, 5],
        'Library': ['AutoGluon', 'Auto-Sklearn', 'Autokeras', 'Pycaret', 'Ludwig'],
        'Runtime(m)': [3, 5, 180, 40, 120],
        'MAE': [0.3456, 0.4208, 0.4716, 0.5273, 0.7149],
        'MSE': [0.2912, 0.3956, 0.4750, 0.5178, 1.0594],
        'R2': [0.8773, 0.8386, 0.7996, 0.779, 0.5692]
    }

    # Erstellen Sie den DataFrame
    df_results = pd.DataFrame(data_for_table)
    tab.subheader("""Top AutoML Performers""")
    tab.write("""In the comparative analysis of various machine learning libraries, AutoGluon and Auto-Sklearn have emerged as the standout performers.
     Not only did they achieve the highest precision in predictions, but they also demonstrated remarkable efficiency in runtime.
       This combination of precision and speed makes them prime choices for those seeking efficient and accurate autoML solutions". For developers without high-powered machines, AutoGluon and Auto-Sklearn are excellent recommendations""")
    tab.table(data_for_table)

    tab.subheader("Auto-Machine Learning for Polymer Property Prediction from SMILES Strings")
    tab.write("In this tutorial, we focus on using Auto-ML libraries to predict polymer properties from SMILES strings, which are used to represent the structure of chemical species. We will generate molecular fingerprints from these SMILES strings, providing a numerical representation that can be used for machine learning predictions.")



    tab1, tab2, tab3, tab4, tab5 = tab.tabs(
        ["AutoGluon", "AutoSklearn", "Pycaret", "Auto-Keras", "Ludwig"])

    auto_gluon_tab.display(tab1)
    auto_sklearn_tab.display(tab2)
    pycaret_tab.display(tab3)
    auto_keras_tab.display(tab4)
    ludwig_tab.display(tab5)