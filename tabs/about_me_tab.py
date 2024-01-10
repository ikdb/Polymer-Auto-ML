# tab1.py
import streamlit as st

def display(tab):
    tab.write("""

    About Me:

    Hi, I'm Ibrahim! 🙋‍♂️ I'm currently pursuing my bachelor's in computer science, and I have a profound passion for data science. 💻 What truly excites me is the practical application of machine learning and data analytics, especially when it's related to polymer research. 🧪 Alongside my studies, I've had the opportunity to immerse myself in professional roles where I delved deep into analyzing the impact behavior of polymer foams. When I'm not engrossed in academics, you can find me analyzing stock market data 📈, indulging in various sports ⚽🏀, or simply staying active. 🏃 I have a voracious appetite for learning about new technological advancements 🚀 and am always on the lookout to expand my horizons. Living a healthy lifestyle is not just a choice, but a commitment for me. 🍏🧘‍♂️

    Always eager to connect and share, I value the conversations and insights this community brings. Whether you're here for the tech chat or just curious about my journey, I'm glad you dropped by. Let's dive into the world of data together!""")
    tab.subheader('Connect with me!')

    # Social media/contact links.
    tab.markdown('Feel free to reach out to me through any of these platforms:')
    link_linked_in = '[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ikdb)'
    link = '[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ibrahim-karademir-227783281)'
    link_telegram = '[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/IbrahimKarademir)'
    link_email = '[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ibrahim.karademir@uni-bayreuth.de)'

    # Defining a row with 3 columns to place the buttons side by side
    col1, col2, col3, col4 = tab.columns(4)  # '4' represents the number of buttons/icons you have.

    # Placing each button in a column. This will display them side by side.
    with col1:
        col1.markdown(link_linked_in, unsafe_allow_html=True)  # GitHub

    with col2:
        col2.markdown(link, unsafe_allow_html=True)  # LinkedIn

    with col3:
        col3.markdown(link_telegram, unsafe_allow_html=True)  # Telegram

    with col4:
        col4.markdown(link_email, unsafe_allow_html=True)  # Email
