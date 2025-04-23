import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px; padding-top: 20px;">
        <h1 style="font-family: 'Arial Black', Gadget, sans-serif; color: #FF6F61; font-size: 48px; margin-bottom: 10px;">🏋️‍♂️ Personal Fitness Tracker</h1>
        <p style="font-size: 18px; font-style: italic; color: #ddd;">"Push yourself because no one else is going to do it for you."</p>
        <hr style="border: 2px solid #FF6F61; width: 50px; margin: 20px auto;">
    </div>
    """,
    unsafe_allow_html=True
)

# Updated name and description as per user request
st.write("## Track Your Fitness Journey and Burn Calories Effectively")
st.write("Unlock the power of your body's energy expenditure with our advanced calorie prediction model. Tailored to your unique profile, this tool helps you understand how your age, gender, BMI, and exercise habits influence your calorie burn.")
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")

st.sidebar.header("User Input Parameters: ")

# Group sidebar inputs with tooltips
def user_input_features():
    st.sidebar.subheader("Personal Info")
    age = st.sidebar.slider("Age: ", 10, 100, 30, help="Select your age in years")
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"), help="Select your gender")

    st.sidebar.subheader("Body Metrics")
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20, help="Body Mass Index")
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38, help="Your body temperature in Celsius")

    st.sidebar.subheader("Exercise Details")
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15, help="Duration of exercise in minutes")
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80, help="Your heart rate during exercise")

    st.sidebar.subheader("Targeted Calories")
    target_calories = st.sidebar.slider("Targeted Calories to Burn: ", 100, 1000, 300, help="Your target calories to burn")

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender,
        "Target_Calories": target_calories
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")

# Display user parameters fully
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Preserve Target_Calories before reindexing
target_calories_value = df["Target_Calories"].values[0]

df = df.reindex(columns=X_train.columns, fill_value=0)

prediction = random_reg.predict(df)

st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.markdown(f"<h2 style='color:#FFD700; font-weight: bold;'>{round(prediction[0], 2)} <small>kilocalories</small></h2>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; margin-top: 30px; margin-bottom: 30px; font-style: italic; color: #FF6F61; font-weight: bolder; font-size: 32px;">
        Keep burning more calories! 💪
    </div>
    """,
    unsafe_allow_html=True
)

# Calorie burnt visualization section below user parameters section
st.subheader("Calories Burned Visualization")

calorie_value = round(prediction[0], 2)
# Use preserved target_calories_value for visualization
target_calories = target_calories_value
max_calories = target_calories if target_calories > 0 else 500  # use target calories or fallback to 500

# Fix for negative or zero remaining calories to avoid pie chart errors
if calorie_value > max_calories:
    calorie_value = max_calories
remaining_calories = max_calories - calorie_value

fig, ax = plt.subplots(figsize=(6, 6))
sizes = [calorie_value, remaining_calories]

# Use vibrant colors for the pie chart slices
colors = ['#FF6F61', '#6BCB77']  # vibrant coral and green

labels = [f'Burned: {calorie_value} kcal', f'Remaining: {remaining_calories} kcal']

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 14, 'weight': 'bold'}
)

for text in texts:
    text.set_color('black')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# Add color legend below the pie chart
st.markdown(
    """
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background-color: #FF6F61; border-radius: 3px;"></div>
            <span>Burned Calories</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background-color: #6BCB77; border-radius: 3px;"></div>
            <span>Remaining Calories</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")
st.header("Similar Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write("---")
st.header("General Information: ")

boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

# Enhance subnotes with markdown and colors
st.markdown(f"You are older than **{round(sum(boolean_age) / len(boolean_age), 2) * 100}%** of other people.")
st.markdown(f"Your exercise duration is higher than **{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}%** of other people.")
st.markdown(f"You have a higher heart rate than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}%** of other people during exercise.")
st.markdown(f"You have a higher body temperature than **{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}%** of other people during exercise.")

# Add footer
st.markdown("---")
st.markdown("<center><small>© 2024 AI-Personal Fitness Tracker. All rights reserved.</small></center>", unsafe_allow_html=True)
