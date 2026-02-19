import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load cleaned data
df = pd.read_csv("cleaned_dataset.csv")

#Features and target
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

print (df.columns)

#Department&EducationalLevel are text columns(machine do not understand- so convert into numbers)
#Remove ID column
df = df.drop("EmployeeID", axis=1)

#Convert categorical columns to numbers
df = pd.get_dummies(df, columns=["Department", "EducationalLevel"], drop_first=True)

print(df.head())