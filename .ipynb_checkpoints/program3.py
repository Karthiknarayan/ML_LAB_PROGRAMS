import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\pshee\Desktop\sem 7\ML LAB\labset\Ml_programs_dataset\dataset\Play Tennis.csv")
df = df.iloc[:, 1:]

# Encode categorical values into integers
str_to_int = preprocessing.LabelEncoder()
df = df.apply(str_to_int.fit_transform)

# Define features and target
col = ["Outlook", "Temprature", "Humidity", "Wind"]
x = df[col]
y = df.Play_Tennis

# Split data into training and testing sets (set random_state for reproducibility)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Train Decision Tree model
classifier = DecisionTreeClassifier(criterion="gini", random_state=100)
classifier.fit(x_train, y_train)

# Make predictions
y_pred = classifier.predict(x_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report and confusion matrix
print(classification_report(y_test, y_pred))

# Create and display a DataFrame with test and predicted values
data_p = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
print(data_p)
