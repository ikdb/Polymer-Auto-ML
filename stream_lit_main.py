import streamlit as st
from tabs import about_me_tab, my_project_tab, polymer_tab, train_your_model_tab, automated_machine_learning_tab  # Importieren der Tab-Module



# Erstellen Sie Ihre Tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs(["My Project", "Polymere", "Train Your Own Model","Automated machine learning", "About Me",])

my_project_tab.display(tab1)
polymer_tab.display(tab2)

train_your_model_tab.display(tab3)

automated_machine_learning_tab.display(tab4)
about_me_tab.display(tab5)
# Sidebar for the LLM Chatbot
with st.sidebar:
    # Header for the chatbot section with emoji
    st.subheader("LLama Chatbot 🦙")

    # Description
    st.write("Ask any questions about polymers, this work, or this website here. Our LLama Chatbot is here to help!")

    # Adding a line for visual separation
    st.markdown("---")

    # Input field for the user's question
    user_input = st.text_input("Type your question here...")

    # Button to send the question to the chatbot
    if st.button("Submit Question"):
        # This space is where the chatbot's response will go.
        # We create a dedicated area for the response to make it visually distinct.
        st.write("The answer to your question will appear here!")  # Placeholder response

    # Adding a line for visual separation after the response area
    st.markdown("---")