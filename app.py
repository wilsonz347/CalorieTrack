import streamlit as st
import pandas as pd
import pickle as pk

def load_model():
    with open('linear_regressor.pkl', 'rb') as f:
        model = pk.load(f)
    return model

def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pk.load(f)
    return scaler

def main():
    st.title('Calories Burned Estimator App')

    st.expander('How It Works').write('This tool uses machine learning to predict how many calories you might burn during a workout. Here\'s how it works: \n1. You provide some information about yourself and your workout.\n2. Our model, trained on data from many different workouts, estimates your calorie burn.')

    st.expander('Methods Used').write('\n1. Data Preparation: Cleaned and processed over a large dataset using Pandas.\n2. Model Selection: Linear Regression model for simplicity.\n3. Training and Validation: Scaled and split the data into a ratio of 80:20 using Scikit-Learn.\n4. Deployment: Employed Streamlit for its simple user interface.')

    st.expander('Transparency').write('For transparency purposes, we leveraged advanced AI-powered technology to enhance our debugging process.')

    st.expander('Credits').write('This project utilizes the Gym Members Exercise Dataset by Vala Khorasani, available under the Apache 2.0 License. The original dataset can be found at https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset.')

    # Load the trained model & scaler
    model = load_model()
    scaler = load_scaler()

    # User input section
    st.header('Enter Your Information')

    age = st.slider('Age', 18, 90, 30)
    gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
    weight = st.slider('Weight (kg)', 40.0, 140.0, 70.0)
    height = st.slider('Height (m)', 1.40, 2.20, 1.70)
    max_bpm = st.slider('Max BPM', 100, 220, 180)
    avg_bpm = st.slider('Average BPM', 60, 200, 120)
    resting_bpm = st.slider('Resting BPM', 40, 100, 60)
    session_duration = st.slider('Session Duration (hours)', 0.25, 3.0, 1.0)
    fat_percentage = st.slider('Fat Percentage', 5.0, 40.0, 20.0)

    # Workout type selector
    workout_type = st.selectbox('Workout Type', ['Cardio', 'HIIT', 'Strength', 'Yoga'])

    # Create a dictionary with all inputs
    input_data = {
        'Age': age,
        'Gender': gender,
        'Weight (kg)': weight,
        'Height (m)': height,
        'Max_BPM': max_bpm,
        'Avg_BPM': avg_bpm,
        'Resting_BPM': resting_bpm,
        'Session_Duration (hours)': session_duration,
        'Fat_Percentage': fat_percentage,
        'Workout_Type_Cardio': 1 if workout_type == 'Cardio' else 0,
        'Workout_Type_HIIT': 1 if workout_type == 'HIIT' else 0,
        'Workout_Type_Strength': 1 if workout_type == 'Strength' else 0,
        'Workout_Type_Yoga': 1 if workout_type == 'Yoga' else 0
    }

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Display the input data
    st.header('Your Input Summary')
    st.write(input_df)

    # Predict the calories burned and display the result
    if st.button('Predict'):
        # Apply scaling to the numerical features
        numerical_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                              'Session_Duration (hours)', 'Fat_Percentage']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        input_array = input_df.values.reshape(1, -1)
        prediction = model.predict(input_array)
        st.success(f"Estimated calories burned: {prediction[0]:.0f} calories")

if __name__ == "__main__":
    main()