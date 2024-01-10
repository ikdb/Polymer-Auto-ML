# tab2.py
import streamlit as st


def display(tab):
    # Section: What are Polymers?
    tab.header("Simple explanation of Automated Machine Learning")

    tab.write("to understand what AutoML is, we should first take a look at what Machine Learning is")

    tab.subheader("Machine learning")
    tab.write("Imagine having a lot of data, for instance, information about people such as their age, height, and weight."
             " ML is like an intelligent detective that identifies patterns in this data to make predictions or solve"
             " problems without us explicitly telling it how to do so. By providing enough examples to the detective,"
             " it can recognize patterns and make decisions, like predicting how tall someone might be based on their"
             " age and weight.")
    #tab.image("")
    tab.write("Machine Learning is a subset of artificial intelligence (AI) that enables computers to identify patterns"
              " in data and learn from them without being explicitly programmed. It relies on algorithms and statistical"
              " models that empower computers to learn from experiences.")

    tab.subheader("Automated machine learning")
    tab.write("""AutoML is like a detective's assistant! Imagine, instead of the detective needing to learn each individual
     method to recognize patterns in data, there's a magical tool that helps them work faster and easier. With AutoML,
      we use computers to automatically discover the best methods for pattern recognition and prediction in data.
      It acts as an intelligent helper guiding the detective without requiring them to start from scratch.
    What's special about AutoML is that it takes on the complex parts of machine learning, making it more accessible 
    for everyone – including non-programmers or experts. AutoML automates model selection, configuration, and optimization
     for machine learning tasks. It automatically identifies the best algorithms and hyperparameters for a specific problem,
      without requiring human intervention or extensive expertise. Through techniques like hyperparameter optimization,
       feature engineering, and model ensembles, AutoML enhances the efficiency of model development.
    AutoML tools expedite the development process by automatically experimenting with various algorithms and configurations
     to create the optimal model for a given dataset. These tools are particularly helpful for users without extensive
      knowledge in machine learning as they simplify and accelerate the modeling process.""")



