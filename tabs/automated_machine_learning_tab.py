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

    tab.image("pictures/Detective.png")

    tab.subheader("Automated machine learning")
    tab.write("""Automated machine learning, or AutoML for short, is like having a magical helper for solving puzzles! Imagine you're trying to find patterns in a jumble of information. Instead of having to figure out every single step on your own, there's this amazing tool that speeds up the process, making it much simpler. AutoML uses computers to automatically find the best ways to spot these patterns and make predictions based on the information given. It's like an intelligent friend who helps guide you without needing to learn everything from the beginning.

What makes AutoML really cool is that it handles the tricky parts of finding patterns in data, which makes it something that almost anyone can use – you don't need to be a computer expert. AutoML takes care of choosing the best methods, setting them up, and making them work as well as possible for spotting patterns and making predictions. It figures out the best approaches and settings for a particular problem all by itself, without a person having to step in or know a lot about the process. By using smart strategies to improve how we develop these methods, AutoML makes the whole process more efficient.

AutoML tools speed up the job by trying out different methods and settings automatically to find the very best one for understanding the information given. These tools are especially great for people who aren't very familiar with the technical side of finding patterns in data because they make everything much easier and quicker to do.""")
    tab.image("pictures/detective assistant.png")


