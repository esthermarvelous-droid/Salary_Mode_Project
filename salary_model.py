import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load cleaned data
df = pd.read_csv("cleaned_dataset.csv")

#Define features and target
X = df[["Age", "MonthlyHoursWorked"]]
y = df["MonthlySalary"]

#Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train model
mode1 = LinearRegression()
mode1.fit(X_train, y_train)

#Predict
predictions = mode1.predict(X_test)

#Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print ("R2 Score:", r2)

#Try Random Forest Model

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest MSE:", rf_mse)
print("random Forest R2:", rf_r2)