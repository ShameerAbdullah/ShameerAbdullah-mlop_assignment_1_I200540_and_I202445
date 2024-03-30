import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("./dataset/data.csv")

y = df['Y']
df = df.drop(columns='Y')

Xt, Xtest, Yt, Ytest = train_test_split(df, y, test_size=0.1, random_state=0)

model = LinearRegression()
model.fit(Xt, Yt)

# Save the trained model to a .pkl file
joblib.dump(model, 'trained_model.pkl')

# print("Model trained and saved successfully!")
