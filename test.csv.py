import streamlit as st
import pandas as pd
import time


def load_csv(input_file):
    # Lesen des Inhalts der hochgeladenen Datei
    df_input = pd.read_excel(input_file)
    return df_input

df_input = load_csv("download_files/housing_price_dataset.csv")
print(df_input.columns.to_list())