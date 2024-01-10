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
from IPython.display import display

# Load the dataset
df = pd.read_csv("/path/to/your/dataset")

#Step 3: Split the Dataset into Training and Testing Sets

# Define the ratio of the dataset to be used for training
# You can change the value of train_ratio to adjust the train-test split ratio
train_ratio = 0.8

# Calculate the split index
split_index = int(len(df) * train_ratio)

# Split the dataset
train_df = df[:split_index]
test_df = df[split_index:]
#Step 4: Prepare the Data and Train the TabularPredictor

# Define the target column
target_column = "your_target_column"

# Separate features and target variable for training and testing sets
X_train = train_df.drop(columns=[target_column]).to_numpy()
y_train = train_df[target_column].to_numpy()

X_test = test_df.drop(columns=[target_column]).to_numpy()
y_test = test_df[target_column].to_numpy()

# Initialize and train the TabularPredictor
predictor = TabularPredictor(label=target_column).fit(train_df)
#Step 6: Evaluate the Model

# Print the best model
print(predictor.get_model_best())

# Evaluate the model on test and training data
predictor.evaluate(test_df)
predictor.evaluate(train_df, silent=True)

#Step 7: Generate and Display the Model Leaderboard

# Generate the model leaderboard
model_leaderboard = predictor.leaderboard(test_df, silent=True)

# Display the model leaderboard
display(model_leaderboard)

# Get the list of models
models = model_leaderboard.model.to_list()

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
    model_overview.loc[len(model_overview)] = [model, evaluation['mean_squared_error'], evaluation['mean_absolute_error'], evaluation['r2'], evaluation['pearsonr'], evaluation['median_absolute_error']]

# Display the model overview
display(model_overview)

#Conclusion: In this tutorial, we have walked through the process of using AutoGluon for tabular data prediction, from installing the necessary libraries to evaluating different models. AutoGluon makes it easy to achieve high-performance models with minimal effort.