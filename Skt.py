import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Read the music dataset
data = pd.read_csv('music.csv.zip')

print("Music Dataset:")  # Display the dataset
print(data)

# Prepare input features (X) and the target variable (Y)
X = data.drop(columns=['genre'])

print("\nInput Features:")  # Display the input features
print(X)

Y = data['genre']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)  # Split the dataset into training and testing sets

# Create and train the decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Make predictions on the test set
prediction = model.predict(X_test)

# Display the predicted genres for test data
results = pd.DataFrame(
    {'Actual Genre': Y_test, 'Predicted Genre': prediction})
print("\nPredicted Genres for Test Data:")
print(results)

# User input for age
user_age = int(input("\nEnter your age: "))

# Prepare input data for the user
user_data = pd.DataFrame({'age': [user_age]})
user_prediction = model.predict(user_data)


print("\nPredicted Genre for Age", user_age, ":")
print(user_prediction[0])
