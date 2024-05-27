import pandas as pd
import pickle

# Load the saved model from file
with open('studyHours.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
df = pd.read_csv('StudyHour-test.csv')

# Independent variable
x = df[['Hours']]


# Output the predictions
predictions = model.predict(x)
print(predictions)

