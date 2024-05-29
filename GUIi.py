#!/usr/bin/env python
# coding: utf-8

# In[24]:


import streamlit as st
import joblib
import numpy as np
from Credit_model import predict_credit_approval


# Load the pre-trained model
model = joblib.load('Decision_tree_model.pkl')  # Change the filename based on your trained model

# Streamlit app
st.title("Credit Card Approval Predictor")
st.write("Fill in the details below to predict credit approval.")

### Identity input form
st.sidebar.header("Identity :")

p_id = st.sidebar.text_input("ID:", "5000000")
# Convert total_income to a number
try:
    total_income = float(p_id)
except ValueError:
    st.error("Please enter a valid number for ID")
gender = st.sidebar.radio("Gender", ["Male", "Female"])

### demographic infos input form
st.sidebar.header("Demographic infos :") 
age = st.sidebar.slider("Age:", 18, 100, 25)
family_status = st.sidebar.selectbox("Marital Status:", ["Single / not married", "Married", "Civil marriage",
                                                         "Separated", "Widow"])
# Total Income entry (as text)
working_years = st.sidebar.text_input("working years", "20")

# Convert total_income to a number (you may want to handle errors here)
try:
    working_years = float(working_years)
except ValueError:
    st.error("Please enter a valid number for Total working years.")


### financial infos input form
st.header("Financial Info: ")
owned_car = st.number_input("Owned Car", min_value=0, value=0)
total_children = st.number_input("Total Children", min_value=0, value=0)
total_family_members = st.number_input("Total Family Members", min_value=1, value=1)
housing_type = st.selectbox("Housing Type", ["House / apartment", "Rented apartment", "With parents", "Office apartment",
                                            "Municipal apartment", "Co-op apartment"])
income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "State servant", "Student"])
total_income = st.slider("Total Income:", 0, 500000, 2000000)

### personal and professional context input form
st.header("Personal and professional context: ")
education_type = st.selectbox("Education Level:", ["Higher education", "Incomplete higher ", "Lower secondary",
                                                      "Secondary / secondary special"])
job_title  = st.selectbox("Job:", ["Accountants", "Cleaning staff", "Cooking staff", "Core staff", "Drivers",
                                          "High skill tech staff", "HR staff", "IT staff", "Laborers", "Low-skill Laborers",
                                          "Managers", "Medicine staff", "Private service staff", "Realty agents", "Sales staff",
                                          "Secretaries", "Security staff", "Waiters/barmen staff"])
owned_realty = st.number_input("Owned Realty", min_value=0, value=0)

### Contacts input form
st.header("Contact Info: ")
owned_mobile_phone = st.number_input("Owned Mobile Phone", min_value=0, value=0)
owned_work_phone = st.number_input("Owned Work Phone", min_value=0, value=0)
owned_phone = st.number_input("Owned Phone", min_value=0, value=0)
owned_email = st.number_input("Owned Email", min_value=0, value=0)
    
### credit history input form
st.header("Credit history: ")  
total_bad_debt = st.number_input("Total Bad Debt", min_value=0, value=0)
total_good_debt = st.number_input("Total Good Debt", min_value=0, value=0)


# Predict button
if st.button("Demand"):
    try:
        # Prepare input data
        input_data = {
            'ID': p_id,
            'gender': gender,
            'age': age,
            'family_status': family_status,
            'total_children': total_children,
            'total_family_members': total_family_members,
            'housing_type': housing_type,
            'owned_car': owned_car,
            'owned_realty': owned_realty,
            'education_type': education_type,
            'job_title': job_title,
            'income_type': income_type,
            'total_income': total_income,
            'working_years': working_years,
            'total_bad_debt': total_bad_debt,
            'total_good_debt': total_good_debt,
            'owned_mobile_phone': owned_mobile_phone,
            'owned_work_phone': owned_work_phone,
            'owned_phone': owned_phone,
            'owned_email': owned_email,
        }

        # Make predictions using the loaded model
        prediction = predict_credit_approval(input_data)
        
        result_text = "Approved" if prediction == 1 else "Not Approved"

        # Display the result
        st.sidebar.success(f"The application is {result_text}.")
    except ValueError as e:
        st.error(str(e))

# Display model details
st.sidebar.header("Model Details:")
st.sidebar.text("Model: Decision Tree Classifier")
st.sidebar.text("Accuracy: 0.93")  # Update with your model's accuracy

# Display dataset information
st.sidebar.header("Dataset Information:")
st.sidebar.text("Number of Instances: 25000")
st.sidebar.text("Features: 21")
st.sidebar.text("Target: Approval Status (1: Approved, 0: Not Approved)")


# In[ ]:




