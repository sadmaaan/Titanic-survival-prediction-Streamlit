import pickle
import numpy as np
import streamlit as st


def load_model(input_data):

    loaded_model = pickle.load(open('titanic_model.sav', 'rb'))

    in_data_nparray = np.asarray(input_data)
    input_reshaped = in_data_nparray.reshape(1, -1)
    prediction_2 = loaded_model.predict(np.asarray(input_reshaped))

    if prediction_2[0] == 1:
        return "The person survived"
    else:
        return "The person didn't survived"


def main():
    
    st.title("Titanic Survival Prediction")
    
    pclass = st.text_input("Enter passenger Pclass")
    sex = st.text_input("Enter passenger sex")
    age = st.text_input("Enter passenger age")
    sibsp = st.text_input("Enter passenger SibSp")
    parch = st.text_input("Enter Parch")
    fare = st.text_input("Enter fare")
    embarked = st.text_input("Enter embarked")
    
    survived = ''
    
    if st.button("Predict"):
        survived = load_model([pclass, sex, age, sibsp, parch, fare, embarked])
        
    st.success(survived)
    
    
if __name__ == '__main__':
    main()