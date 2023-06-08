import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox

window = tk.Tk()
window.title("Music Recommendation System")

data = pd.read_csv(
    'D://VSCode//Python//Music_Recommendation_System//music.csv.zip')

# Prepare the input features (X) and the target variable (Y)
X = data.drop(columns=['genre'])
Y = data['genre']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)


def predict_genre():
    age = int(age_entry.get())
    gender = gender_var.get()
    user_data = pd.DataFrame({'age': [age], 'gender': [gender]})
    user_prediction = model.predict(user_data)
    messagebox.showinfo(
        "Prediction", f"The predicted genre is: {user_prediction[0]}")  # Display the predicted genre


# Create labels and entries
age_label = tk.Label(window, text="Age:")
age_label.pack()
age_entry = tk.Entry(window)
age_entry.pack()

gender_label = tk.Label(window, text="Gender:")
gender_label.pack()
gender_var = tk.StringVar()
gender_radio_male = tk.Radiobutton(
    window, text="Male", variable=gender_var, value="1")
gender_radio_male.pack()
gender_radio_female = tk.Radiobutton(
    window, text="Female", variable=gender_var, value="0")
gender_radio_female.pack()

predict_button = tk.Button(window, text="Predict", command=predict_genre)
predict_button.pack()

window.mainloop()
