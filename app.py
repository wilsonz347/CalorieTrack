import time
import streamlit as st
import pandas as pd
import pickle as pk
import plotly.graph_objects as go
import plotly.express as px

def load_model():
    with open('linear_regressor.pkl', 'rb') as f:
        model = pk.load(f)
    return model

def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pk.load(f)
    return scaler

def main():
    st.set_page_config(
        page_title="CalorieTrack",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    page = st.sidebar.selectbox("Choose a page", ["Project Overview","Exploring Data","Visualization","Prediction"])

    if page == "Project Overview":
        st.title('Calories Burned Estimator App')
        st.header('Project Overview')
        st.expander('How It Works').write('This tool uses machine learning to predict how many calories you might burn during a workout. Here\'s how it works: \n1. You provide some information about yourself and your workout.\n2. Our model, trained on data from many different workouts, estimates your calorie burn.')
        st.expander('Methods Used').write('\n1. Data Preparation: Cleaned and processed over a large dataset using Pandas.\n2. Model Selection: Linear Regression model for simplicity.\n3. Training and Validation: Scaled and split the data into a ratio of 80:20 using Scikit-Learn.\n4. Deployment: Employed Streamlit for its simple user interface.')
        st.expander('Transparency').write('For transparency purposes, we leveraged advanced AI-powered technology to enhance our debugging process.')
        st.expander('Credits').write('This project utilizes the Gym Members Exercise Dataset by Vala Khorasani, available under the Apache 2.0 License. The original dataset can be found at https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset.')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", "900+ entries")
        with col2:
            st.metric("Model Accuracy", "98%")

        st.subheader("Key Features")
        st.checkbox("Personalized calorie burn estimates", value=True, disabled=True)
        st.checkbox("Easy-to-use interface", value=True, disabled=True)
        st.checkbox("Based on real workout data", value=True, disabled=True)

        if 'feedback' not in st.session_state:
            st.session_state.feedback = []

        # Use a unique key for the text area
        if 'feedback_text' not in st.session_state:
            st.session_state.feedback_text = ""

        st.subheader("We'd love your feedback!")

        feedback = st.text_area("Please share your thoughts or suggestions:",
                                value = st.session_state.feedback_text,
                                key="feedback_input",
                                placeholder="Type your feedback here...")

        if st.button("Submit Feedback"):
            if feedback:
                st.session_state.feedback.append(feedback)
                st.session_state.feedback_text = ""
                st.success("Thank you for your feedback!")
            else:
                st.warning("Please enter some feedback before submitting.")

    if page == "Exploring Data":
        st.header('Data Overview')
        st.write('This dataset contains information about 900+ gym members, including their age, gender, weight, height, maximum heart rate, average heart rate, resting heart rate, session duration, workout type, fat percentage, water intake, workout frequency, experience level, and body mass index (BMI).')

        # Preview first five rows
        st.subheader('Data Preview')
        df = pd.read_csv("gym_members_exercise_tracking.csv")
        st.write(df.head())

        # Basic Statistics
        st.subheader('Basic Statistics')
        st.write(df.describe())

        # Data Download
        st.subheader('Download Data')
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='gym_members__exercise_tracking.csv',
            mime='text/csv',
        )

    if page == "Visualization":
        st.subheader('Data Visualizations')

        df = pd.read_csv("gym_members_exercise_tracking.csv")
        # Distribution of Age
        st.write("### Age Distribution")
        fig_age = px.histogram(df, x="Age", nbins=20, title="Distribution of Age", color_discrete_sequence=['#6A0DAD'], opacity=0.7)
        fig_age.update_layout(
            xaxis_title_text='Age',
            yaxis_title_text='Count',
            bargap=0.1,  # Gap between bars
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Add the average age
        mean_age = df['Age'].mean()
        fig_age.add_vline(x=mean_age, line_dash="dash", line_color="#FF4136",
                          annotation_text=f"Mean Age: {mean_age:.1f}",
                          annotation_position="top right")

        # Customize axes
        fig_age.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0',
                             showline=True, linewidth=2, linecolor='#333333')
        fig_age.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0',
                             showline=True, linewidth=2, linecolor='#333333')

        # Add hover information
        fig_age.update_traces(hovertemplate='Age: %{x}<br>Count: %{y}')

        st.plotly_chart(fig_age, use_container_width=True)

        # Distribution of Gender
        st.write("### Gender Distribution")
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']

        fig_gender = px.pie(gender_counts,
                            names='Gender',
                            values='Count',
                            title="Distribution of Gender",
                            color_discrete_sequence=px.colors.qualitative.Set3)

        # Add hover information
        fig_gender.update_traces(hovertemplate='Gender: %{label}<br>Count: %{value}')

        st.plotly_chart(fig_gender, use_container_width=True)

        # Distribution of Experience Level
        st.write("### Experience Level Distribution")
        exp_counts = df['Experience_Level'].value_counts().sort_index()

        exp_df = pd.DataFrame({'Experience_Level': exp_counts.index, 'Count': exp_counts.values})

        fig_exp = px.bar(exp_df,
                         x='Experience_Level',
                         y='Count',
                         color='Experience_Level',
                         title="Distribution of Experience Level",
                         labels={'Experience_Level': 'Experience Level', 'Count': 'Number of Members'},
                         color_discrete_map={1: 'blue', 2: 'green', 3: 'red'})

        fig_exp.update_layout(showlegend=False)

        # Display the chart
        st.plotly_chart(fig_exp)

    # Load the trained model & scaler
    model = load_model()
    scaler = load_scaler()

    # User input section
    if page == "Prediction":
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
            status = st.empty()

            with st.spinner('Calculating...'):
                status.text('Preparing data...')
                progress_bar = st.progress(0)
                for i in range(33):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # Apply scaling to the numerical features
                status.text('Scaling features...')
                numerical_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                                      'Session_Duration (hours)', 'Fat_Percentage']
                input_df[numerical_features] = scaler.transform(input_df[numerical_features])
                for i in range(33, 66):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                status.text('Making prediction...')
                input_array = input_df.values.reshape(1, -1)
                prediction = model.predict(input_array)
                for i in range(66, 100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

            status.text('Prediction complete!')
            st.success(f"Estimated calories burned: {prediction[0]:.0f} calories")

            col1, col2 = st.columns(2)

            with col1:
                # Create radar chart
                st.header('Your Fitness Profile')

                # Select features for the radar chart
                radar_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                                  'Session_Duration (hours)', 'Fat_Percentage']

                # Normalize the data for the radar chart
                radar_data = input_df[radar_features].iloc[0]
                radar_data_normalized = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

                # Create the radar chart using Plotly
                fig = go.Figure(data=go.Scatterpolar(
                    r=radar_data_normalized.values,
                    theta=radar_features,
                    fill='toself'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=False
                )

                # Display the radar chart
                st.plotly_chart(fig)

            with col2:
                # Display actual values
                st.subheader('Your Fitness Metrics')
                for feature in radar_features:
                    original_value = input_data[feature]
                    st.write(f"{feature}: {original_value}")

if __name__ == "__main__":
    main()