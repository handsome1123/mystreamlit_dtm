# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 10:44:19 2025

@author: LAB
"""
# Import libraries
import streamlit as st
import numpy as np
import pickle

# Load model
try:
    with open('dtm_trained_model.pkl', 'rb') as f:
        dtm_model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Application title
st.title("Iris Flower Classification")
st.write("Enter the features of the Iris flower:")

# Input fields
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = dtm_model.predict(input_data)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"The predicted species is: **{species[prediction[0]]}**")
