"""Title: AutoGluon Tutorial for Tabular Data

Introduction: In this tutorial, we will explore how to use AutoGluon for tabular data prediction. AutoGluon automates machine learning tasks, enabling you to easily achieve high-performance models.

Step 1: Install Necessary Libraries
"""
"""%pip install pandas
%pip install autogluon
%pip install ipython
Step 2: Import Libraries and Load Dataset
"""
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np


def predict_data(model, df, prediction_df, target):
    # Überprüfen, ob die Spaltennamen übereinstimmen (außer der Zielvariable)
    df_columns = set(df.columns)
    pred_columns = set(prediction_df.columns)

    if df_columns == pred_columns:
        # Alle Spalten sind gleich, außer das Zielattribut
        if target in prediction_df.columns:
            prediction_df = prediction_df.drop(columns=[target])


        # Durchführen der Vorhersage
        predictions = model.predict(prediction_df)

        predicted_df = prediction_df.copy()
        predicted_df['predicted_target'] = predictions
        return predicted_df

    elif target in prediction_df.columns:
        # Das Zielattribut ist vorhanden, aber andere Spalten fehlen
        prediction_df = prediction_df.drop(columns=[target])

        # Durchführen der Vorhersage
        predictions = model.predict(prediction_df)

        # Erstellen eines neuen DataFrame mit den Vorhersagen
        predicted_df = prediction_df.copy()
        predicted_df['predicted_target'] = predictions
        return predicted_df

    else:
        return "Column names do not match, and the target attribute is not present."

def train_model(df,target_column):

    train_ratio = 0.8

    # Calculate the split index
    split_index = int(len(df) * train_ratio)

    # Split the dataset
    train_df = df[:split_index]
    test_df = df[split_index:]
    #Step 4: Prepare the Data and Train the TabularPredictor

    # Define the target column


    # Separate features and target variable for training and testing sets
    X_train = train_df.drop(columns=[target_column]).to_numpy()
    y_train = train_df[target_column].to_numpy()

    X_test = test_df.drop(columns=[target_column]).to_numpy()
    y_test = test_df[target_column].to_numpy()

    # Initialize and train the TabularPredictor
    predictor = TabularPredictor(label=target_column).fit(train_df)
    #Step 6: Evaluate the Model

    # Print the best model
    #print(predictor.get_model_best())

    # Evaluate the model on test and training data
    predictor.evaluate(test_df)
    predictor.evaluate(train_df, silent=True)

    #Step 7: Generate and Display the Model Leaderboard

    # Generate the model leaderboard
    model_leaderboard = predictor.leaderboard(test_df, silent=True)

    # Display the model leaderboard
    #display(model_leaderboard)

    # Get the list of models
    models = model_leaderboard.model.to_list()
    best_model = models[0]
    #Step 8: Make Predictions and Evaluate Each Model

    # Make predictions
    y_pred = predictor.predict(train_df.drop(columns=[target_column]))

    # Initialize the model overview DataFrame
    columns = ['model', 'mean_squared_error', 'mean_absolute_error', 'r2', 'pearsonr', 'median_absolute_error']
    model_overview = pd.DataFrame(columns=columns)

    # Evaluate each model and add the results to the model overview DataFrame
    for model in models:
        predictor.set_model_best(model)
        evaluation = predictor.evaluate(test_df)
        model_overview.loc[len(model_overview)] = [model, abs(evaluation['mean_squared_error']), abs(evaluation['mean_absolute_error']), evaluation['r2'], evaluation['pearsonr'], evaluation['median_absolute_error']]

    # Display the model overview
    predictor.set_model_best(best_model)
    return predictor,model_overview


