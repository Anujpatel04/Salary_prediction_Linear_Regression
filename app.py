import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open(r'C:\Users\a\VSCODE_NAREDH-IT\MACHINE-LEARNING\Salary_pridict_project\linear_regression_model.pkl', 'rb'))

# Streamlit app title
st.title('Salary Prediction')

# Instruction
st.write('Enter the years of experience to predict the salary')

# Input for years of experience
years_of_experience = st.number_input(
    'Enter years of experience', 
    min_value=0.0, 
    max_value=100.0, 
    value=0.0, 
    step=0.5
)

# Predict button
if st.button('Predict Salary'):
    # Prepare the input data
    input_data = np.array([[years_of_experience]])
    # Predict the salary
    salary = model.predict(input_data)
    # Display the predicted salary
    st.success(f"The predicted salary for {years_of_experience} years of experience is ${salary[0]:,.2f}")

# Footer
st.write('Thank you for using the app! made by anuj patel')
