def display(tab):
    tab.subheader("Auto-Sklearn Tutorial")

    # Beschreibung aktualisieren
    tab.write("""
    In this tutorial, we explore the application of Auto-Sklearn for predicting outcomes based on tabular data, showcasing its capability to automate the machine learning process. Developed as part of the AutoML project at the University of Freiburg, Auto-Sklearn abstracts the complexity involved in model training, offering an accessible and efficient tool for both professionals and enthusiasts.

    Auto-Sklearn excels in automated machine learning tasks such as hyperparameter tuning, model selection, and feature engineering, drastically reducing the time required to develop robust machine learning models. With its advanced features and integration capabilities, Auto-Sklearn is a prominent tool in the AutoML library ecosystem.
    """)
    tab.subheader("Step 1: Install Necessary Libraries")
    tab.code("""
    %pip install pandas
    %pip install numpy
    %pip install autosklearn
    %pip install scikit-learn
    """)

    tab.subheader("Step 2: Importing Libraries and Loading Data")
    tab.code("""
    import pandas as pd
    import numpy as np
    from psmiles import PolymerSmiles as PS
    from autosklearn.regression import AutoSklearnRegressor

    # Read csv and filter EGC
    egc_df = pd.read_csv("path/to/export.csv")
    egc_df = egc_df[egc_df.property == "Egc"]
    """)

    tab.subheader("Step 3: Preprocessing and Feature Engineering")
    tab.code("""
    smile_string_list = list(egc_df.smiles.values)
    value_list = list(egc_df.value.values)

    finger_print_list = []
    for ss in smile_string_list:
        current_polymer = PS(ss)
        finger_print_list.append(current_polymer.fingerprint())

    fp_list_formatted = [fp.tolist() for fp in finger_print_list]

    ss_fp_egc_df = pd.DataFrame({
        'Smile': smile_string_list,
        'Finger_prints': fp_list_formatted,
        'Egc': value_list
    })
    print(ss_fp_egc_df.head())
    """)

    tab.subheader("Step 4: Preparing the Dataset")
    tab.code("""
    # Assuming that the fingerprint data is numeric and suitable for model input
    X = np.array(fp_list_formatted)
    y = np.array(value_list)

    # Split the dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """)

    tab.subheader("Step 5: Model Training")
    tab.code("""
    # Initialize AutoSklearn regressor
    autosklearn_regressor = AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        n_jobs=-1
    )
    autosklearn_regressor.fit(X_train, y_train)

    # Display the best models found by AutoSklearn
    print(autosklearn_regressor.leaderboard())
    """)

    tab.subheader("Step 6: Evaluate the Model")
    tab.code("""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    train_predictions = autosklearn_regressor.predict(X_train)
    test_predictions = autosklearn_regressor.predict(X_test)

    # Calculate metrics for training data
    r2_train = r2_score(y_train, train_predictions)
    print("R2 for training data:", r2_train)

    # Calculate metrics for testing data
    r2_test = r2_score(y_test, test_predictions)
    print("R2 for test data:", r2_test)
    """)
