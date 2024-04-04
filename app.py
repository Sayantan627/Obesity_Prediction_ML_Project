import streamlit as st
import os
import pickle
import streamlit as st
# from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd


# Load the machine learning model
mp = pickle.load(open('model_pickle.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set the layout of the app to 'wide'

st.set_page_config(page_title="Obesity prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# Function to determine the color based on obesity prediction
def get_color(obesity_prediction):
    if obesity_prediction < 18.5:
        return 'red'
    elif 18.5 <= obesity_prediction < 25.0:
        return 'yellow'
    elif 25.0 <= obesity_prediction < 30.0:
        return 'yellow'  # You can adjust the colors as needed
    elif 30.0 <= obesity_prediction < 35.0:
        return 'orange'
    elif 35.0 <= obesity_prediction < 40.0:
        return 'orange'  # You can adjust the colors as needed
    else:
        return 'red'  # For obesity III and higher

# Sidebar
obesity_prediction = 28.0  # Example value, replace it with your actual prediction
color = get_color(obesity_prediction)

st.sidebar.markdown(f'<h3 style="color:{color};">Obesity Prediction</h3>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="color:{color};">Underweight: Less than 18.5</p>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="color:{color};">Normal: 18.5 to 24.9</p>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="color:{color};">Overweight: 25.0 to 29.9</p>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="color:{color};">Obesity I: 30.0 to 34.9</p>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="color:{color};">Obesity II: 35.0 to 39.9</p>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p style="color:{color};">Obesity III: Higher than 40</p>', unsafe_allow_html=True)

st.title('Obesity Prediction Web App')

# MAIN PART

# CONTAINER- 1 : personal info

# Create a container for the personal information inputs
with st.container(border=True):
    st.header('Personal Information')

    # Display input parameters for personal information
    col1, col2, col3 = st.columns([1,2,2] , gap='large')

    with col1:
        st.subheader('Gender')
        gender = st.selectbox('', df['Gender'].unique())

    with col2:
        st.subheader('Age')
        age = st.number_input('Enter your age', min_value=5, max_value=100)

    with col3:
        st.subheader('Height (cm)')
        height = st.number_input('Enter your height')

    # col4, col5 = st.columns([1,2],gap="large")
    with st.container():
        st.subheader('Weight (kg)')
        weight = st.number_input('Enter your weight')

    with st.container():
        st.subheader('Family history of overweight')
        family_overweight = st.selectbox('', df['Family History with Overweight'].unique())



# CONTAINER -2 : eating habits

# Create a container for the eating habits inputs
with st.container(border=True):
    st.header('Eating Habits')

    # Display input parameters for eating habits
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Frequent consumption of high caloric food (FAVC)')
        favc = st.selectbox('', df['Frequent consumption of high caloric food'].unique())

    with col2:
        st.subheader('Frequency of consumption of vegetables (FCVC)')
        fcvc = st.selectbox('', df['Frequency of consumption of vegetables'].unique(), key='fcvc_key')

    with st.container():
        st.subheader('Number of main meals (NCP)')
        ncp = st.number_input('Enter the number of main meals', min_value=1, max_value=10)

    with st.container():
        st.subheader('Consumption of food between meals (CAEC)')
        caec = st.selectbox('', df['Consumption of food between meals'].unique())

    with st.container():
        st.subheader('Consumption of water daily (CH20)')
        ch20 = st.selectbox('', df['Consumption of water daily'].unique())

    with st.container():
        st.subheader('Consumption of alcohol (CALC)')
        calc = st.selectbox('', df['Consumption of alcohol'].unique())

    with st.container():
        st.subheader('Smoking')
        smoke = st.selectbox('', df['Smoke'].unique(), key='smoke_key')






# CONTAINER 3 - Physical Condition Attributes

with st.container(border=True):
    st.header('Physical Condition Attributes')

    # Divide the container into two columns
    col1, col2 = st.columns(2)

    with col1:
        # Physical activity frequency (FAF)
        st.subheader('Physical Activity Frequency (FAF)')
        faf_key = 'faf_key'  # Unique key for this radio button
        faf = st.selectbox('', df['Physical activity frequency'].unique(), key=faf_key)

        # Calories consumption monitoring (SCC)
        st.subheader('Calories Consumption Monitoring (SCC)')
        scc_key = 'scc_key'  # Unique key for this radio button
        scc = st.selectbox('', df['Calories consumption monitoring'].unique(), key=scc_key)

    with col2:
        # Transportation used (MTRANS)
        st.subheader('Transportation Used (MTRANS)')
        mtrans_key = 'mtrans_key'  # Unique key for this select box
        mtrans = st.selectbox('', df['Transportation used'].unique(), key=mtrans_key)

        # Time using technology devices (TUE)
        st.subheader('Time Using Technology Devices (TUE)')
        tue_key = 'tue_key'  # Unique key for this radio button
        tue = st.selectbox('', df['Time using technology devices'].unique(), key=tue_key)



# Function to process input data and convert it into a numpy array
def process_input_data(gender, age, height, weight, family_overweight, favc, fcvc, ncp, caec, ch20, calc, smoke, faf, scc, mtrans, tue):

    # Convert categorical data to numerical data
    gender_female = 1 if gender == 'Female' else 0
    gender_male = 1 if gender == 'Male' else 0
    family_overweight_no = 1 if family_overweight == 'no' else 0
    family_overweight_yes = 1 if family_overweight == 'yes' else 0
    favc_no = 1 if favc == 'no' else 0
    favc_yes = 1 if favc == 'yes' else 0
    caec_always = 1 if caec == 'Always' else 0
    caec_frequently = 1 if caec == 'Frequently' else 0
    caec_sometimes = 1 if caec == 'Sometimes' else 0
    caec_no = 1 if caec == 'no' else 0
    smoke_no = 1 if smoke == 'no' else 0
    smoke_yes = 1 if smoke == 'yes' else 0
    scc_no = 1 if scc == 'no' else 0
    scc_yes = 1 if scc == 'yes' else 0
    calc_always = 1 if calc == 'Always' else 0
    calc_frequently = 1 if calc == 'Frequently' else 0
    calc_sometimes = 1 if calc == 'Sometimes' else 0
    calc_no = 1 if calc == 'no' else 0
    mtrans_automobile = 1 if mtrans == 'Automobile' else 0
    mtrans_bike = 1 if mtrans == 'Bike' else 0
    mtrans_motorbike = 1 if mtrans == 'Motorbike' else 0
    mtrans_public_transportation = 1 if mtrans == 'Public Transportation' else 0
    mtrans_walking = 1 if mtrans == 'Walking' else 0
    
    
    # Convert input data to a numpy array
    input_data = np.array([age, height, weight, fcvc, ncp, ch20, faf, tue, gender_female, gender_male, family_overweight_no, family_overweight_yes, favc_no, favc_yes, caec_always, caec_frequently, caec_sometimes, caec_no, smoke_no, smoke_yes, scc_no, scc_yes, calc_always, calc_frequently, calc_sometimes, calc_no, mtrans_automobile, mtrans_bike, mtrans_motorbike, mtrans_public_transportation, mtrans_walking])
    
    return input_data.reshape(1, -1)


# Create a button for the prediction result
if st.button("Predict"):
    # Make a prediction using the user input and the machine learning model
    input_data = process_input_data(gender, age, height, weight, family_overweight, favc, fcvc, ncp, caec, ch20, calc, smoke, faf, scc, mtrans, tue)
    prediction = mp.predict(input_data)

    prediction_label = ''


    if(prediction[0] == 0):
        prediction_label = "Underweight"
    elif(prediction[0] == 1):
        prediction_label = "Normal Weight"
    elif(prediction[0] == 2):
        prediction_label = "Obesity Type I"
    elif(prediction[0] == 3):
        prediction_label = "Obesity Type II"
    elif(prediction[0] == 4):
        prediction_label = "Obesity Type III"
    elif(prediction[0] == 5):
        prediction_label = "Overweight Level I"
    else:
        prediction_label = "Overweight Level II"

    st.write('Prediction:', prediction_label)

    # # Display the prediction result on the result page
    # display_result_page(prediction)