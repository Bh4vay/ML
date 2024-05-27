from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('StudyHour.csv')

# Separate the dataset into features (x) and target variable (y)
x = df[['Hours']]
y = df['Scores']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=10)

# Initialize the Linear Regression model
lr = LinearRegression()

# Train the model with the training data
model = lr.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = model.score(x_test, y_test)
print(f'Accuracy is: {accuracy*100:.2f}%')

# Save the trained model to a file using pickle
with open('studyHours.pkl', 'wb') as f:
    pickle.dump(model, f)
