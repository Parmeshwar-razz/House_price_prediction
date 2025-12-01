# House_price_prediction
🏡 California Housing Price Prediction

A machine learning project that predicts house prices using the California Housing Dataset.
The workflow includes EDA, outlier treatment, feature selection, scaling, model training, and saving the model using pickle.

📌 Project Overview

This project uses the California Housing dataset from scikit-learn to build a Linear Regression model that predicts median house prices.

Key steps performed:

Loaded California Housing dataset

Converted data into a pandas DataFrame

Performed outlier capping using quantiles

Removed highly correlated & unnecessary columns

Scaled features using StandardScaler

Trained a Linear Regression model

Saved the trained model as House_pred.pkl

📊 Technologies Used
Tool / Library	Purpose
Python	Core language
NumPy	Numerical operations
Pandas	Data cleaning & manipulation
Seaborn / Matplotlib	Data visualization
Scikit-learn	Machine learning model & preprocessing
Pickle	Saving the trained model
📂 Project Structure
├── House_pred.pkl           # Saved ML model
├── housing_model.ipynb      # Notebook (if used)
├── main.py                  # Main Python script (optional)
├── README.md                # Documentation
└── requirements.txt         # Dependencies file

🛠️ Steps Performed in Code
1️⃣ Load Dataset

Fetched California Housing dataset using:

from sklearn.datasets import fetch_california_housing

2️⃣ Create DataFrame

Converted dataset into pandas DataFrame and inspected features.

3️⃣ Outlier Treatment

Applied winsorization (capping) on these columns:

Population

HouseAge

AveBedrms

AveOccup

Using:

data["Population"] = data["Population"].clip(lower=low, upper=high)

4️⃣ Drop Unnecessary Features

Removed:

Latitude — highly correlated with Longitude

AveRooms — redundant after ratio features

5️⃣ Add Target Column
data["Price"] = housing.target

6️⃣ Train–Test Split
x_train , X_test , y_train , y_test = train_test_split(x, y, test_size=0.3)

7️⃣ Feature Scaling
scaler = StandardScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(X_test)

8️⃣ Model Training
LR = LinearRegression()
LR.fit(x_train_norm, y_train)

9️⃣ Save Model Using Pickle
pickle.dump(LR, open("House_pred.pkl", "wb"))

📈 Model Used
🔹 Linear Regression

A simple, interpretable model suitable for continuous price prediction.

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Python Script
python main.py

3️⃣ Use the Saved Model

You can load the model in any application:

import pickle
model = pickle.load(open("House_pred.pkl", "rb"))

🎯 Future Improvements

Possible upgrades:

Train more models (Random Forest, XGBoost, etc.)

Deploy the model using Flask, FastAPI, or Streamlit

Add hyperparameter tuning

Add feature engineering & model comparison

🙌 Author

Parmeshwar Rajpurohit
Aspiring Data Scientist & ML Enthusiast
