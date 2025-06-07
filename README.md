# Personal Fitness Tracker

## Description
Personal Fitness Tracker is a web application built with Streamlit that allows users to track their fitness journey by predicting the number of calories burned during exercise. The app uses machine learning models trained on real-world exercise and calorie datasets to provide personalized calorie burn predictions based on user input parameters such as age, gender, BMI, exercise duration, heart rate, and body temperature.

## Features
- User-friendly interface to input personal and exercise parameters.
- Predicts calories burned using a Random Forest regression model.
- Visualizes calories burned versus targeted calories with an interactive pie chart.
- Displays similar exercise data samples based on predicted calorie range.
- Provides general information comparing user parameters with dataset statistics.
- Includes progress bars for interactive feedback during prediction and data display.

## Installation
1. Clone the repository.
2. Ensure Python 3.x is installed.
3. Install required packages:
   ```
   pip install streamlit numpy pandas matplotlib seaborn scikit-learn plotly
   ```
4. Place `calories.csv` and `exercise.csv` in the project directory.

## Usage
Run the Streamlit app:
```
streamlit run app.py
```
Use the sidebar to input your personal info, body metrics, and exercise details. The app will display your parameters, predict calories burned, and show visualizations and related data.

## Dataset Description
- `calories.csv`: Contains user IDs and corresponding calories burned.
- `exercise.csv`: Contains user IDs and exercise-related features such as gender, age, height, weight, duration, heart rate, and body temperature.
- The datasets are merged and preprocessed to calculate BMI and prepare features for the prediction model.

## Model Details
- The app uses a Random Forest Regressor trained on the merged dataset.
- Features used include gender, age, BMI, duration, heart rate, and body temperature.
- The model predicts the calories burned during exercise based on user inputs.

## Visualization
- Pie chart showing burned vs remaining calories relative to the user's targeted calories.
- Data table showing similar exercise records within a close calorie range.
- Statistical comparisons of user parameters against the dataset.

## License
This project is licensed under the MIT License.

© 2024 AI-Personal Fitness Tracker. All rights reserved.


## Author
[Bejjanki Sri Vardhan]
