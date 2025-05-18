import streamlit as st
import pandas as pd
import pickle

with open('ipl_score_predict_model.pkl', 'rb') as f:
    model, feature_columns = pickle.load(f)

teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
         'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
         'Delhi Daredevils', 'Sunrisers Hyderabad']

st.title("IPL Score Predictor")
st.markdown("Predict the final score based on match conditions.")

bat_team = st.selectbox('Select Batting Team', teams)
bowl_team = st.selectbox('Select Bowling Team', [team for team in teams if team != bat_team])

overs_input = st.number_input('Overs Completed (min 5.0)', min_value=5.0, max_value=20.0, step=0.1, format="%.1f")
full_overs = int(overs_input)
balls = round((overs_input - full_overs) * 10)
if balls >= 6:
    full_overs += 1
    balls = 0
overs = full_overs + balls / 6.0

runs = st.number_input('Current Runs', min_value=0)
wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10)
runs_last_5 = st.number_input('Runs in Last 5 Overs', min_value=0)
wickets_last_5 = st.number_input('Wickets in Last 5 Overs', min_value=0)

if st.button('Predict Score'):
    input_df = pd.DataFrame([{
        'bat_team': bat_team,
        'bowl_team': bowl_team,
        'overs': overs,
        'runs': runs,
        'wickets': wickets,
        'runs_last_5': runs_last_5,
        'wickets_last_5': wickets_last_5
    }])
    input_df = pd.get_dummies(input_df, columns=['bat_team', 'bowl_team'])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Final Score: {int(prediction)}")
