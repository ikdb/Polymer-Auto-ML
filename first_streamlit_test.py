import streamlit as st
import pandas as pd

tab1, tab2, tab3,tab4 = st.tabs(["Train your Model", "Auto ML Project","Polymer","About Me"])
tab1.header("Fast AI")
tab1.write("\n Train your own Machine Learning Model")

tab1.subheader("1. Upload your csv or excel file")


with tab1.expander("Data format"):
    tab1.write(
        "The dataset should be a tabular dataset and can contain any number of columns.")

input = tab1.file_uploader('')

if input is not None:
    # Überprüfen Sie den Dateityp
    if input.type == "text/csv":
        tab1.success("Successfully uploaded!")
    elif input.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        tab1.write("Successfully uploaded!")
    else:
        tab1.write("Please upload a CSV or Excel file")





tab2.header("Fast AI")
tab2.write("\n My Project ")
tab2.write("""
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
tab2.subheader("""Top AutoML Performers""")
tab2.write("""In the comparative analysis of various machine learning libraries, AutoGluon and Auto-Sklearn have emerged as the standout performers.
 Not only did they achieve the highest precision in predictions, but they also demonstrated remarkable efficiency in runtime.
   This combination of precision and speed makes them prime choices for those seeking efficient and accurate autoML solutions". For developers without high-powered machines, AutoGluon and Auto-Sklearn are excellent recommendations""")
tab2.table(data_for_table)

auto_gluon_tab,auto_sklearn_tab,pycaret_tab,auto_keras_tab,ludwig_tab = tab2.tabs(["AutoGluon","AutoSklearn","Pycaret","Auto-Keras","Ludwig"])
auto_gluon_tab.write("""Title: AutoGluon Tutorial for Tabular Data
In this tutorial, we delve into the application of AutoGluon for predicting outcomes based on tabular data, showcasing its unique ability to simplify the often complex world of machine learning. Developed by AWS Labs, AutoGluon takes the grunt work out of model training, providing an accessible and efficient tool for professionals and enthusiasts alike.

What sets AutoGluon apart is its advanced automation capabilities, notably in hyperparameter tuning, model selection, and feature engineering, significantly reducing the time typically required to produce a robust machine learning model. Furthermore, its seamless integration and scalability features, backed by AWS's powerful cloud infrastructure, allow it to stand out in the diverse ecosystem of AutoML libraries.
Step 1: Install Necessary Libraries""")

auto_gluon_tab.code("""%pip install pandas
%pip install autogluon
%pip install ipython""")

auto_gluon_tab.write("""Step 2: Import Libraries and Load Dataset""")
auto_gluon_tab.code("""from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
from IPython.display import display

# Load the dataset
df = pd.read_csv("/path/to/your/dataset")""")

auto_gluon_tab.write("""Step 3: Split the Dataset into Training and Testing Sets""")
auto_gluon_tab.code("""# Define the ratio of the dataset to be used for training
# You can change the value of train_ratio to adjust the train-test split ratio
train_ratio = 0.8

# Calculate the split index
split_index = int(len(df) * train_ratio)

# Split the dataset
train_df = df[:split_index]
test_df = df[split_index:]""")
auto_gluon_tab.write("""Step 4: Prepare the Data and Train the TabularPredictor""")
auto_gluon_tab.code("""# Define the target column
target_column = "your_target_column"

# Separate features and target variable for training and testing sets
X_train = train_df.drop(columns=[target_column]).to_numpy()
y_train = train_df[target_column].to_numpy()

X_test = test_df.drop(columns=[target_column]).to_numpy()
y_test = test_df[target_column].to_numpy()

# Initialize and train the TabularPredictor
predictor = TabularPredictor(label=target_column).fit(train_df)""")
auto_gluon_tab.write("""Step 6: Evaluate the Model""")
auto_gluon_tab.code("""# Print the best model
print(predictor.get_model_best())

# Evaluate the model on test and training data
predictor.evaluate(test_df)
predictor.evaluate(train_df, silent=True)""")
auto_gluon_tab.write("""Step 7: Generate and Display the Model Leaderboard""")
auto_gluon_tab.code("""# Generate the model leaderboard
model_leaderboard = predictor.leaderboard(test_df, silent=True)

# Display the model leaderboard
display(model_leaderboard)

# Get the list of models
models = model_leaderboard.model.to_list()""")
auto_gluon_tab.write("""Step 8: Make Predictions and Evaluate Each Model""")
auto_gluon_tab.code("""# Make predictions
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
auto_gluon_tab.subheader("Conclusion")
auto_gluon_tab.write("""In this tutorial, we have walked through the process of using AutoGluon for tabular data prediction, from installing the necessary libraries to evaluating different models. AutoGluon makes it easy to achieve high-performance models with minimal effort.""")
# Finden Sie den Pfad zur aktuellen Datei, damit Sie darauf relative Pfade aufbauen können.
auto_gluon_path = "download_files/Auto_Gluon_tutorial.ipynb"
auto_gluon_path_py = "download_files/Auto_Gluon_tutorial.py"
# Öffnen Sie die Datei im Binärmodus.
col1,col2 = auto_gluon_tab.columns(2)
with open(auto_gluon_path, "rb") as file:
    bytes_data = file.read()
col1.download_button(label="Download Jupyter Notebook",data=bytes_data,file_name="Auto_Gluon_tutorial.ipynb",mime="application/octet-stream")
with open(auto_gluon_path_py, "rb") as file:
    bytes_data_py = file.read()
col2.download_button(label="Download Python file",data=bytes_data_py,file_name="Auto_Gluon_tutorial.py",mime="application/octet-stream")



#tab2.echo()

tab4.write("""

About Me:

Hi, I'm Ibrahim! 🙋‍♂️ I'm currently pursuing my bachelor's in computer science, and I have a profound passion for data science. 💻 What truly excites me is the practical application of machine learning and data analytics, especially when it's related to polymer research. 🧪 Alongside my studies, I've had the opportunity to immerse myself in professional roles where I delved deep into analyzing the impact behavior of polymer foams. When I'm not engrossed in academics, you can find me analyzing stock market data 📈, indulging in various sports ⚽🏀, or simply staying active. 🏃 I have a voracious appetite for learning about new technological advancements 🚀 and am always on the lookout to expand my horizons. Living a healthy lifestyle is not just a choice, but a commitment for me. 🍏🧘‍♂️

Always eager to connect and share, I value the conversations and insights this community brings. Whether you're here for the tech chat or just curious about my journey, I'm glad you dropped by. Let's dive into the world of data together!""")
tab4.subheader('Connect with me!')

# Social media/contact links.
tab4.markdown('Feel free to reach out to me through any of these platforms:')
link_linked_in = '[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ikdb)'
link = '[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ibrahim-karademir-227783281)'
link_telegram = '[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/IbrahimKarademir)'
link_email = '[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ibrahim.karademir@uni-bayreuth.de)'


# Defining a row with 3 columns to place the buttons side by side
col1, col2, col3, col4 = tab4.columns(4)  # '4' represents the number of buttons/icons you have.

# Placing each button in a column. This will display them side by side.
with col1:
    col1.markdown(link_linked_in, unsafe_allow_html=True)  # GitHub

with col2:
    col2.markdown(link, unsafe_allow_html=True)  # LinkedIn

with col3:
    col3.markdown(link_telegram, unsafe_allow_html=True)  # Telegram

with col4:
    col4.markdown(link_email, unsafe_allow_html=True)  # Email




