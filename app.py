# streamlit web app

#Import Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load train model
model=tf.keras.models.load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)
with open('one_hot_encoding.pkl','rb') as file:
    one_hot_encoding=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# streamlit app
st.title('Customer churn prediction')

#User input
geography=st.selectbox('Geography', one_hot_encoding.categories_[0])
gender=st.selectbox('Gender', label_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Memebr',[0,1])

# preparing input data
input_df=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded=one_hot_encoding.transform([[geography]])
geo_encoded_df=pd.DataFrame(geo_encoded.toarray(), columns=one_hot_encoding.get_feature_names_out(['Geography']))
input_df=pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1)

input_scaled=scaler.transform(input_df)

prediction=model.predict(input_scaled)
st.write(f'prediction_prob:{prediction[0][0]:.2f}')
if(prediction[0][0]>0.5):
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')