import pandas as pd
import streamlit as st

from utilities import prediction_pipeline

DEFAULT = 0

def main():
    st.title("Bank Customer Churn Predictor")
    
    # Collecting user inputs
    Surname = st.text_input("Customer Surname")
    CreditScore = st.number_input("Customer Credit Score", min_value=0, max_value=1000, format="%d")
    Geography = st.selectbox("Choose customer geography", ("Spain", "France", "Germany"))
    Gender = st.radio("Enter Customer Gender", ("Male", "Female"))
    Age = st.number_input("Customer Age", min_value=0, format="%d")
    Tenure = st.number_input("Customer Tenure", min_value=0, max_value=10, format="%d")
    Balance = st.number_input("Customer Balance", min_value=1.0, format="%f")
    NumOfProducts = st.selectbox("Choose which product did the customer use", (1, 2, 3, 4))
    HasCrCard = 1 if st.radio("Did The Customer Have a Credit Card?", ('Yes', 'No')) == 'Yes' else 0
    IsActiveMember = 1 if st.radio("Is The Customer an Active Member?", ('Yes', 'No')) == 'Yes' else 0
    EstimatedSalary = st.number_input("Customer Estimated Salary", min_value=1.0, format="%f")

    # Creating a data dictionary
    data = {'RowNumber': DEFAULT,
            'CustomerId': DEFAULT,
            'Surname': Surname,
            'CreditScore': CreditScore,
            'Geography': Geography,
            'Gender': Gender,
            'Age': Age,
            'Tenure': Tenure,
            'Balance': Balance,
            'NumOfProducts': NumOfProducts,
            'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember,
            'EstimatedSalary': EstimatedSalary}
    
    # Converting data dictionary to DataFrame
    data_new = pd.DataFrame(data, index=[0])
    
    # Predicting customer churn
    if st.button('Predict!'):
        output = prediction_pipeline(data_new)
        if output[0] == 1:
            st.write('{} Likely Exited The Bank (Churn)'.format(Surname))
        else:
            st.write('{} Likely NOT Exited The Bank (No Churn)'.format(Surname))

if __name__ == "__main__":
    main()
