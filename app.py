import streamlit as st 
import pickle
import numpy as np 

# load the saved model 
model = pickle.load(open(r"C:\Users\adder\Desktop\Data Science\salary prediction\linear_regression_model.pkl",'rb')) 

# set the title of the streamlit app
st.title("Salary Prediction App")

# add a brief description 
st.write("This app predict the salary based on the years of experience using a simple linear regression")


# add input widget for usere to enter years of experience 
years_experience = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, value=1.0, step = 0.5)

# when the button is clicked make predictions
if st.button("Predict Salary"):
    experience_input = np.array([[years_experience]])
    Prediction = model.predict(experience_input)
    
    
    # Display the result
    st.success(f"The predicted salary for {years_experience} years of experience is: ${Prediction[0]:,.2f}")

# Display information about the model 
st.write("The model was trained using a dataset of salary and years of experience.")