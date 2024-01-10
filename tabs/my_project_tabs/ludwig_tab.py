# tab1.py
import streamlit as st

def display(tab):
    tab.subheader("Ludwig")

    
    tab.write("""
           In this tutorial, we delve into the application of AutoGluon for predicting outcomes based on tabular data, showcasing its unique ability to simplify the often complex world of machine learning. Developed by AWS Labs, AutoGluon takes the grunt work out of model training, providing an accessible and efficient tool for professionals and enthusiasts alike.

           What sets AutoGluon apart is its advanced automation capabilities, notably in hyperparameter tuning, model selection, and feature engineering, significantly reducing the time typically required to produce a robust machine learning model. Furthermore, its seamless integration and scalability features, backed by AWS's powerful cloud infrastructure, allow it to stand out in the diverse ecosystem of AutoML libraries.
           Step 1: Install Necessary Libraries
           """)
    tab.subheader("Step 1: Install Necessary Libraries")

    tab.code("""
    %pip install pandas
    %pip install ludwig
    %pip install sklearn
    %pip install ipython
    %pip install numpy""")
    tab.subheader("Step 2: Importing Libraries and Loading Data")
    tab.code("""
    from psmiles import PolymerSmiles as PS
    import pandas as pd
    import numpy as np
    from ludwig.automl import auto_train
    from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

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
    
auto_train_results = auto_train(
    dataset=train_df,
    target=target_column,
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)""")

    tab.subheader("Step 6: Evaluate Model")
    tab.code("""
    best_model = auto_train_results.best_model
    predictions = best_model.predict(test_df)


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test,predictions)
print("mae: ",mae," mse: ", mse," r2: ", r2)
    
    
    """)

