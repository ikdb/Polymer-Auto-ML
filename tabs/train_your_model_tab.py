import streamlit as st
import pandas as pd
import time
from io import BytesIO
import auto_gluon_auto_ml

prediction = False
model_trained = False
model = None
model_overview = None
show_button = False
df = 0
predicted = False
def load_csv(input_file):
    # Lesen des Inhalts der hochgeladenen Datei
    df_input = pd.read_csv(input_file)
    return df_input

def load_uploaded_file(input_file):
    # Abhängig vom Dateityp die entsprechende Funktion aufrufen
    df = 0
    if input_file.type == "application/vnd.ms-excel" or input_file.type == "text/csv":
        df = load_csv(input_file)
    elif input_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(input_file)  # Für Excel-Dateien
    show_button = True

    return df, show_button

def get_column_names_and_types(dataframe):
    columns = dataframe.columns.tolist()  # Spaltennamen als Liste erhalten
    types = [dataframe[col].dtype for col in columns]  # Datentypen der Spalten erhalten
    return columns, types

def convert_text_to_df(text,existing_df,column_names,predicted_column):
    values = text.split(', ')
    column_index = column_names.index(predicted_column)
    values.insert(column_index, None)
    # Hinzufügen der neuen Zeile zum vorhandenen DataFrame
    existing_df = existing_df.append(pd.Series(values, index=column_names), ignore_index=True)

    # Die neueste Zeile im DataFrame abrufen
    new_row = existing_df.iloc[-1]

    # Entfernen der neuesten Zeile aus dem DataFrame
    existing_df = existing_df.iloc[:-1]

    # Erstellen eines neuen DataFrames mit nur der neuen Zeile
    new_df = pd.DataFrame([new_row], columns=column_names)
    new_df = new_df.drop(columns=[predicted_column])

    return new_df





def display(tab):

    global model_trained,prediction,model,model_overview,show_button,df,predicted


    tab.subheader("Train your own Machine Learning Model")

    tab.subheader("1. Upload your csv or excel file.")
    with tab.expander("Data format"):## nur einen upload erlauben
        st.write(
            "The dataset should be a tabular dataset and can contain any number of columns.")

    # Hochladen der Datei

    with st.spinner('Uploading file...'):
        input_file = tab.file_uploader('Upload Data', type=["csv", "xlsx"])

    if input_file is not None:
        #with st.spinner('Loading data..'):
        try:
            df,show_button = load_uploaded_file(input_file)
            tab.success("Successfully uploaded!")
        except Exception as e:
            st.write(f"An error occurred: {e}")

    tab.subheader("2. Choose your target column")
    tab.write("Please select the column from your data that you'd like the model to predict. This is your 'target' or 'label' column. It should contain the outcome or result that you're investigating.")

    if show_button:
        target_columns = df.columns.tolist()
        choice = tab.selectbox("target", target_columns)
        if choice:
            tab.write(f"You selected {choice} as the target column.")

    tab.subheader("3. Train your model")
    #model_trained = False
    if show_button:
        if tab.button("Start"):
            with st.spinner('Training in progress...'):
                time.sleep(2)  # Simulieren einer Verzögerung für das Training
                model_trained = True  # Setzen Sie dies auf True, wenn das Training erfolgreich war
                model,model_overview = auto_gluon_auto_ml.train_model(df=df,target_column=choice)
                tab.success("Training completed!")

    tab.subheader("4. See your results")
    # Erklärung für den Benutzer hinzufügen, was in diesem Abschnitt erwartet wird
    if model_trained:
        # Hier würden Sie Ergebnisse anzeigen, Diagramme zeichnen, Metriken anzeigen usw.
        tab.write(
            "Here you will see the predictions made by your model, along with various performance metrics and visualizations that give you insights into your model's effectiveness. Since your model has been trained, the results are now available.")
        tab.table(model_overview)
        with tab.expander("Explanation of Evaluation Metrics"):
            st.write("### R² (Coefficient of Determination)")
            st.write("R² measures how well the dependent variable is explained by the predictions.")
            st.write("Evaluation:")
            st.write("Perfect: 1 | Excellent: > 0.9 | Good: > 0.8 | Acceptable: > 0.7 | Moderate: > 0.5 | Weak: <= 0.5")

            st.write("### RMSE (Root Mean Square Error)")
            st.write("RMSE measures the average deviation of predictions from actual values.")
            st.write("Lower RMSE values indicate better accuracy.")
            st.write("The closer to 0, the better the model performance.")

            st.write("### MSE (Mean Squared Error)")
            st.write("MSE measures the average squared difference between predictions and actual values.")
            st.write("Evaluation:")
            st.write("The closer to 0, the better the model performance.")
            st.write("A value of 0 would indicate perfect predictions.")
            st.write("Larger values indicate larger errors.")

        with tab.expander("Potential Issues with Categorical Data"):
            st.write("### Common Problems with Categorical Data")

            st.write(
                "1. **Need for Numbers:** Categorical data, like words or types, need to be turned into numbers for the computer to understand them. Sometimes this change might be tricky!")

            st.write(
                "2. **Too Few Examples:** If there are not enough examples in the data, like not enough rows or things to learn from, the computer might have trouble finding patterns.")

            st.write(
                "3. **Lots of Different Types:** When there are many different categories that don’t seem similar, the computer might get confused and find it hard to learn from them. For example, if the categories are too different from each other, it could make it tough for the computer to understand how they relate.")

            st.write(
                "4. **Missing or Rare Types:** If some categories are missing or don’t happen very often, the computer might not know what to do when it sees them.")

            st.write(
                "It’s important to help the computer understand the categories properly before using AutoGluon. This could mean turning words into numbers or making sure there are enough examples for the computer to learn from!")

    else:
        tab.write(
            "Once your model has been trained by clicking the 'Start' button above, the results will be displayed here. This section will provide insights into the model's performance, including predictions, performance metrics, and potentially visualizations of the model's effectiveness.")

    tab.subheader("5. Predict Data")

    if model_trained:
        # Hier würden Sie Ergebnisse anzeigen, Diagramme zeichnen, Metriken anzeigen usw.
        with tab.expander("Data format for Predictions"):
            st.write(
                """
                Please format your data for predictions using one of the following methods:

                1. **Comma-separated values :** Provide the column values in the correct order, separated by commas. For example:
                   ```
                   column1_value, column2_value, column3_value, ...
                   ```

                2. **JSON or dictionary format:** Submit your data in JSON format with key-value pairs corresponding to column names and their values. For example:
                   ```json
                   {
                       "column1": "value",
                       "column2": "value",
                       "column3": "value",
                       ...
                   }
                   ```

                3. **File upload:** You may also upload a complete CSV or Excel file containing all the necessary columns corresponding to the dataframe structure.

                Ensure that your data is clean and properly structured to avoid any processing errors.
                """)


        input_prediction_text = tab.text_input("Input data")
        #tab.write('The current movie title is', input_prediction_text)
        """enter_clicked = tab.button("Submit")

        # Wenn der Button "Submit" gedrückt wurde und der Input nicht leer ist
        if enter_clicked:
            print("enter")
            column_names, column_types = get_column_names_and_types(df)
            input_prediction_df = convert_text_to_df(input_prediction_text,df, column_names)
            print(input_prediction_df)"""
        #if input_prediction_text != None:
        #    column_names, column_types = get_column_names_and_types(df)
        #    input_prediction_df = convert_text_to_df(input_prediction_text, df, column_names,choice)
        #    print(input_prediction_df)
        prediction_df = 0
        with st.spinner('Uploading file...'):
            input_file_prediction = tab.file_uploader('Upload Data For Predictions', type=["csv", "xlsx"],key=1)
        if input_file_prediction is not None:
            prediction_df, prediction = load_uploaded_file(input_file_prediction)
            #prediction = True

        if prediction == True:
            if tab.button("Predict"):

                prediction_df = auto_gluon_auto_ml.predict_data(model=model,df=df,prediction_df=prediction_df,target=choice)
                predicted = True
            if predicted == True:
                if isinstance(prediction_df, pd.DataFrame):
                    # Wenn es ein DataFrame ist, zeige eine Vorschau und füge einen Download-Button hinzu
                    tab.write("Preview of the data:")
                    tab.table(prediction_df.head(5))  # Zeige die ersten fünf Zeilen des DataFrames an
                    file_name_predictions = f"{choice}_prediction.xlsx"
                    with BytesIO() as buffer:
                        prediction_df.to_excel(buffer, index=False, engine='openpyxl')
                        tab.download_button('Download Predictions', buffer.getvalue(), file_name_predictions)

            elif isinstance(prediction_df, str):
                    tab.write(prediction_df)






        #tab.download_button("Download your model",model)
    else:
        tab.write(
            "Once your model has been trained you can predict values")


