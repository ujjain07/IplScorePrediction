import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from datetime import datetime

df = pd.read_csv('ipl.csv')
df.drop(labels=['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'], axis=1, inplace=True)

consistent_team = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                   'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                   'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_team)) & (df['bowl_team'].isin(consistent_team))]
df = df[df['overs'] >= 5.0]
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = pd.get_dummies(df, columns=['bat_team', 'bowl_team'])

X_train = df[df['date'].dt.year <= 2016].drop(labels=['total', 'date'], axis=1)
X_test = df[df['date'].dt.year >= 2017].drop(labels=['total', 'date'], axis=1)
y_train = df[df['date'].dt.year <= 2016]['total'].values
y_test = df[df['date'].dt.year >= 2017]['total'].values

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print("MAE:", metrics.mean_absolute_error(y_test, prediction))
print("MSE:", metrics.mean_squared_error(y_test, prediction))
print("R2 Score:", metrics.r2_score(y_test, prediction) * 100)

with open('ipl_score_predict_model.pkl', 'wb') as f:
    pickle.dump((model, list(X_train.columns)), f)
