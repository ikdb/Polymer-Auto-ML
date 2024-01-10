
import streamlit as st

def display(tab):
    tab.subheader("AutoGluon")

    tab.write("""
       In this tutorial, we delve into the application of AutoGluon for predicting outcomes based on tabular data, showcasing its unique ability to simplify the often complex world of machine learning. Developed by AWS Labs, AutoGluon takes the grunt work out of model training, providing an accessible and efficient tool for professionals and enthusiasts alike.

       What sets AutoGluon apart is its advanced automation capabilities, notably in hyperparameter tuning, model selection, and feature engineering, significantly reducing the time typically required to produce a robust machine learning model. Furthermore, its seamless integration and scalability features, backed by AWS's powerful cloud infrastructure, allow it to stand out in the diverse ecosystem of AutoML libraries.
       Step 1: Install Necessary Libraries
       """)
    tab.subheader("Step 1: Install Necessary Libraries")

    tab.code("""
%pip install pandas
%pip install autogluon
%pip install ipython
%pip install numpy""")
    tab.subheader("Step 2: Importing Libraries and Loading Data")
    tab.code("""
from psmiles import PolymerSmiles as PS
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

# Read csv and filter EGC
egc_df = pd.read_csv("../recources/export.csv")
egc_df = egc_df[egc_df.property == "Egc"]
            """)

    tab.subheader("Step 3: Generating Fingerprints from SMILES Strings")
    tab.code("""
            #creating fingerprints for SS
smile_string_list = list(egc_df.smiles.values)
value_list = list(egc_df.value.values)
finger_print_list = []
for smile_string in smile_string_list:
    curren_polymere = PS(smile_string)
    finger_print_list.append(curren_polymere.fingerprint())


fp_list_formatted = []
for i in finger_print_list:
    fp_list_formatted.append(i.tolist())
    
print(len(fp_list_formatted[0]))
ss_fp_egc_df = pd.DataFrame({'Smile': smile_string_list, 'Finger_prints': fp_list_formatted, 'Egc': value_list})
print(ss_fp_egc_df)



final_df = pd.DataFrame(
    np.array(finger_print_list), index=smile_string_list
)

ss = pd.Series(value_list, index=smile_string_list, name="Egc")

concatenated_df = pd.concat(
    [final_df, ss],
    axis=1,
).reset_index(names="psmiles")
print(concatenated_df.set_index("psmiles").columns)



auto_ml_df = concatenated_df.sample(frac=1, random_state=0)#shulles the df
print(auto_ml_df.shape)


psmlies_column = auto_ml_df["psmiles"]
auto_ml_df = auto_ml_df.drop(columns=["psmiles"])
auto_ml_df.reset_index(drop=True, inplace=True)
print(auto_ml_df.shape)
            """)

    tab.subheader("Step 4: Split the Dataset into Training and Testing Sets")
    tab.code("""
            split_index = int(len(auto_ml_df) * 0.8)

train_df = auto_ml_df[:split_index]
test_df = auto_ml_df[split_index:]
""")

    tab.subheader("Step 5: Prepare the Data and Train the Model")
    tab.code("""
X_train = train_df.drop(columns=[target_column]).to_numpy()
y_train = train_df[target_column].to_numpy()

X_test = test_df.drop(columns=[target_column]).to_numpy()
y_test = test_df[target_column].to_numpy()

# Initialize and train the TabularPredictor
target_column = "Egc
predictor = TabularPredictor(label=target_column).fit(train_df)""")


    tab.subheader("""Step 6: Evaluate the Model""")
    tab.code("""
#Print the best model
print(predictor.get_model_best())

# Evaluate the model on test and training data
predictor.evaluate(test_df)
predictor.evaluate(train_df, silent=True)""")
    tab.subheader("""Step 7: Generate and Display the Model Leaderboard""")
    tab.code("""
# Generate the model leaderboard
model_leaderboard = predictor.leaderboard(test_df, silent=True)

# Display the model leaderboard
display(model_leaderboard)

# Get the list of models
models = model_leaderboard.model.to_list()""")
    tab.write("""Step 8: Make Predictions and Evaluate Each Model""")
    tab.code("""
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
display(model_overview)""")
    tab.subheader("Conclusion")
    tab.write(
        """In this tutorial, we have walked through the process of using AutoGluon for tabular data prediction, from installing the necessary libraries to evaluating different models. AutoGluon makes it easy to achieve high-performance models with minimal effort.""")
    # Finden Sie den Pfad zur aktuellen Datei, damit Sie darauf relative Pfade aufbauen können.
    auto_gluon_path = "download_files/Auto_Gluon_tutorial.ipynb"
    auto_gluon_path_py = "download_files/Auto_Gluon_tutorial.py"
    # Define your Google Colab URL
    google_colab_url = "https://colab.research.google.com/drive/1SwAjYexsVOtiQ7svw1QZspZoc0I1S81J?usp=sharing"

    # Öffnen Sie die Datei im Binärmodus.
    col1, col2, col3 = tab.columns(3)
    with open(auto_gluon_path, "rb") as file:
        bytes_data = file.read()
    col1.download_button(label="Download Jupyter Notebook", data=bytes_data, file_name="Auto_Gluon_tutorial.ipynb",
                         mime="application/octet-stream")
    with open(auto_gluon_path_py, "rb") as file:
        bytes_data_py = file.read()
    col2.download_button(label="Download Python file", data=bytes_data_py, file_name="Auto_Gluon_tutorial.py",
                         mime="application/octet-stream")

    # Erstelle einen Button in der dritten Spalte, der die Nutzer zu Google Colab weiterleitet
    col3.link_button('Open in Google Colab', google_colab_url)
