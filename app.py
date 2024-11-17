import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

model = pickle.load(open(r'C:\Users\a\VSCODE_NAREDH-IT\MACHINE-LEARNING\Salary_pridict_project\linear_regression_model.pkl', 'rb'))

st.title('Salary Prediction')

st.write('Enter the years of experience to predict the salary')

years_of_experience = st.number_input(
    'Enter years of experience', 
    min_value=0.0, 
    max_value=100.0, 
    value=0.0, 
    step=0.5
)

if st.button('Predict Salary'):
    input_data = np.array([[years_of_experience]])
    salary = model.predict(input_data)
    st.success(f"The predicted salary for {years_of_experience} years of experience is ${salary[0]:,.2f}")

st.write('Thank you for using the app! made by anuj patel')
