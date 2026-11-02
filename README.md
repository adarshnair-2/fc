# fc

5th---------
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Load
df = pd.read_csv("/content/drive/MyDrive/CS2225 Datasets for
lab/Program5_StudentsPerformance.csv")
X = df[['reading score']] # input
y = df['math score'] # output
# Split, train, predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
# Evaluate
print("Slope :", round(model.coef_[0], 3))
print("Intercept :", round(model.intercept_, 3))
print("MSE :", round(mean_squared_error(y_test, y_pred), 2))
print("R2 score :", round(r2_score(y_test, y_pred), 3))
# Predict a new value
new = pd.DataFrame({'reading score': [70]})
print("Predicted math score for reading score 70:", round(model.predict(new)[0], 2))


6th---------
