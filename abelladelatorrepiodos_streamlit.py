import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

with open('RandomForest_model.pkl', 'rb') as file:
    RClassifier = pickle.load(file)

# Define the Streamlit app
def main():

    st.title('Obesity Classification App')

    # Sidebar for user input
    st.sidebar.header('User Input')
    age = st.sidebar.slider('Age', 14, 100, step=1)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    height = st.sidebar.slider('Height (cm)', 100, 300, step=1)
    weight = st.sidebar.slider('Weight (kg)', 20, 500, step=1)

    # Calculate BMI
    bmi = weight / (height / 100) ** 2
    gender_encoded = 1 if gender == 'Female' else 0

    # Button for prediction
    if st.sidebar.button('Predict'):
        # Make prediction
        prediction = RClassifier.predict([[age, gender_encoded, height, weight, bmi]])

        # Display prediction
        st.subheader('Prediction')
        st.write(f'The predicted label is: {prediction[0]}')

if __name__ == '__main__':
    main()
