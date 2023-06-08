import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the music dataset
data = pd.read_csv(
    'D://VSCode//Python//Music_Recommendation_System//music.csv.zip')


print("Music Dataset:")  # Display the dataset
print(data)


# Prepare the input features (X) and the target variable (Y)
X = data.drop(columns=['genre'])


print("\nInput Features:")  # Display the input features
print(X)

Y = data['genre']


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)  # Split the dataset into training and testing sets
model = DecisionTreeClassifier()  # Create and train the decision tree classifier
model.fit(X_train, Y_train)

# Make predictions on the test set
predictions = model.predict(X_test)


results = pd.DataFrame(
    {'Actual Genre': Y_test, 'Predicted Genre': predictions})  # Display the predicted genres for test data
print("\nPredicted Genres for Test Data:")
print(results)


user_age = int(input("\nEnter the age: "))  # User input for age and gender
user_gender = input("Enter the gender (M/F): ")

# Prepare input data for the user
user_data = pd.DataFrame({'age': [user_age], 'gender': [user_gender]})

# Make prediction for the user input
user_prediction = model.predict(user_data)


# Display the predicted genre for the user input
print("\nPredicted Genre :")
print(user_prediction[0])
