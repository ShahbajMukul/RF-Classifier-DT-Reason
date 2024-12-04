# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

st.title('Machine Downtime Prediction')

# Load model and metadata with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load('Z4Model.pkl')
        feature_names = pd.read_csv('feature_names.csv')['features'].tolist()
        target_names = pd.read_csv('target_names.csv')['targets'].tolist()
        return model, feature_names, target_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all required files are present.")
        return None, None, None

model, feature_names, target_names = load_model()

if model is None:
    st.stop()

# Input form
st.header('Enter Machine Details')

machine_type = st.selectbox(
    'Machine Type',
    ['Sidel Blowmolder', 'Krones Blowmolder', 'Husky Injection Molder']
)

location = st.selectbox(
    'Machine Location',
    ['Downtown', 'Pontiac', 'Westland']
)

duration = st.number_input('Duration (minutes)', min_value=10, max_value=240)
start_time = st.number_input('Start Time (minutes from midnight)', min_value=0, max_value=1440)
end_time = start_time + duration

if st.button('Predict'):
    try:
        # Create input data and transform (same as before)
        input_data = pd.DataFrame({
            'MachineType': [machine_type],
            'MachineLocation': [location],
            'Duration': [duration],
            'StartTime': [start_time],
            'EndTime': [end_time]
        })

        # One-hot encode categorical variables
        for col in ['MachineType', 'MachineLocation']:
            encoder = joblib.load(f'{col}_encoder.pkl')
            encoded = encoder.transform(input_data[[col]])
            encoded_df = pd.DataFrame(
                encoded, 
                columns=encoder.get_feature_names_out([col])
            )
            input_data = pd.concat([input_data, encoded_df], axis=1)
            input_data.drop(columns=[col], inplace=True)

        # Reorder columns
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Get probabilities for each target
        probabilities = model.predict_proba(input_data)
        group_probs = {}
        code_probs = {}

        # Iterate over each target's probabilities
        for i, proba in enumerate(probabilities):
            target = target_names[i]  # Assuming target_names is ordered
            class_labels = model.classes_[i]
            # Map class labels to probabilities
            class_probabilities = dict(zip(class_labels, proba[0]))
            # Add to group or code probabilities
            if target.startswith('Group_'):
                group_probs[target.replace('Group_', '')] = class_probabilities[1]
            elif target.startswith('Code_'):
                code_probs[target.replace('Code_', '')] = class_probabilities[1]

        # Sort probabilities
        group_probs = dict(sorted(group_probs.items(), key=lambda x: x[1], reverse=True))
        code_probs = dict(sorted(code_probs.items(), key=lambda x: x[1], reverse=True))

        # Create visualizations
        st.header('Prediction Probabilities')

         # Print results to console
                 # Print results to console
        print("Top 2 Group Probabilities:", list(group_probs.items())[:2])
        print("Top 2 Code Probabilities:", list(code_probs.items())[:2])


        # Group probabilities plot
        fig_group = go.Figure(data=[
            go.Bar(
                x=list(group_probs.values()),
                y=list(group_probs.keys()),
                orientation='h',
                marker_color='lightblue'
            )
        ])
        fig_group.update_layout(
            title='Group Probabilities',
            xaxis_title='Probability',
            yaxis_title='Group',
            height=400
        )
        st.plotly_chart(fig_group)

        # Code probabilities plot
        fig_code = go.Figure(data=[
            go.Bar(
                x=list(code_probs.values()),
                y=list(code_probs.keys()),
                orientation='h',
                marker_color='lightgreen'
            )
        ])
        fig_code.update_layout(
            title='Code Probabilities',
            xaxis_title='Probability',
            yaxis_title='Code',
            height=600  # Taller to accommodate more codes
        )
        st.plotly_chart(fig_code)

        # Display top 3 most likely predictions
        st.subheader('Top 3 Most Likely Predictions:')
        
        st.write("Groups:")
        for group, prob in list(group_probs.items())[:3]:
            st.write(f"- {group}: {prob:.2%}")
        
        st.write("Codes:")
        for code, prob in list(code_probs.items())[:3]:
            st.write(f"- {code}: {prob:.2%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")