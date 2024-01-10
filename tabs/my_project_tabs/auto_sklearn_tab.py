# tab1.py
import streamlit as st

def display(tab):
    ####autosklearn###

    tab.header("AutoSklearn")
    tab.write("""
        Auto-machine learning frameworks are designed to automate the process of applying machine learning algorithms. These frameworks optimize machine learning pipelines using various techniques such as Bayesian optimization, meta-learning, and ensemble learning. This approach is particularly beneficial for applying high-quality machine learning models without an extensive background in the field or the need for extensive trial-and-error tuning.

        In this tutorial, we will explore the application of an auto-ML framework to predict polymer properties from SMILES strings, which encode the structure of chemical species. Our approach will include generating molecular fingerprints from these SMILES strings to provide a numerical representation that can be utilized in machine learning predictions.
        """)

    tab.subheader("Step 1: Install Necessary Libraries")
    tab.code("""
        %pip install pandas
        %pip install psmiles
        %pip install scikit-learn
        %pip install auto-ml
        """)

    tab.subheader("Step 2: Preparing the Dataset")
    tab.code("""
        import pandas as pd
        from psmiles import PolymerSmiles as PS

        # Load your dataset
        data = pd.read_csv("/path/to/your/dataset.csv")

        # Filter for a specific property, for example 'Egc'
        data = data[data.property == "Egc"]
        """)

    tab.subheader("### Step 3: Generating Fingerprints from SMILES Strings")
    tab.code("""
        # Create fingerprints for each polymer based on the SMILES strings
        fingerprints = []
        for smile in data['smiles']:
            polymer = PS(smile)
            fingerprint = polymer.fingerprint()
            fingerprints.append(fingerprint)

        # Add fingerprints to the dataframe
        data['fingerprints'] = fingerprints
        """)

    tab.subheader("Step 4: Preparing the Data for Auto-ML")
    tab.code("""
        # Prepare the data for training
        X = list(data['fingerprints'])
        y = data['value'].values  # Assuming 'value' is the column with property measurements
        """)

    tab.write("### Step 5: Training with an Auto-ML Framework")
    tab.code("""
        from auto_ml import AutoMLRegressor  # Hypothetical import, replace with actual library

        # Initialize the AutoML regressor
        regressor = AutoMLRegressor(
            time_left_for_this_task=120, 
            per_run_time_limit=30
        )

        # Train the model
        regressor.fit(X, y)
        """)

    tab.write("### Step 6: Model Evaluation")
    tab.code("""
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

        # Predict on the test set
        y_pred = regressor.predict(X_test)

        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae}, MSE: {mse}, R^2: {r2}")
        """)

    tab.write("""
        By following these steps, you can undertake a comprehensive process for predicting the properties of polymers from their SMILES strings using an auto-ML framework. This methodology leverages the power of automated machine learning to significantly simplify the predictive modeling process.
        """)

    # Instructions to download necessary files for local setup, Google Colab, and Python script
    tab.write("### Download the Tutorial Files")
    # Add your download buttons and links here...
    # End of Auto-ML Tutorial
    # pycaret


