import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle


#  LOAD DATA

housing = fetch_california_housing()
print(housing.feature_names)
print('-' * 80)
print(housing.data)
print('-' * 80)
print(housing.DESCR)

data = pd.DataFrame(housing.data, columns=housing.feature_names)

# Check missing values
data.isnull().sum()

# Display first few rows
data.head(10)


#  OUTLIER TREATMENT (CAP OUTLIERS)

# Population
low = data["Population"].quantile(0.02)
high = data["Population"].quantile(0.99)
data["Population"] = data["Population"].clip(lower=low, upper=high)

# HouseAge
low_age = data["HouseAge"].quantile(0.02)
high_age = data["HouseAge"].quantile(0.99)
data["HouseAge"] = data["HouseAge"].clip(lower=low_age, upper=high_age)

# AveBedrms
low_bed = data["AveBedrms"].quantile(0.02)
high_bed = data["AveBedrms"].quantile(0.99)
data["AveBedrms"] = data["AveBedrms"].clip(lower=low_bed, upper=high_bed)

# AveOccup
low_occ = data["AveOccup"].quantile(0.02)
high_occ = data["AveOccup"].quantile(0.99)
data["AveOccup"] = data["AveOccup"].clip(lower=low_occ, upper=high_occ)

#  DROP UNNECESSARY / CORRELATED COLUMNS

data.drop("Latitude", axis=1, inplace=True)     # High correlation with Longitude
data.drop("AveRooms", axis=1, inplace=True)    # Not needed after room ratio features

#  ADD TARGET COLUMN

data["Price"] = housing.target

#  BOXPLOT TO VERIFY OUTLIERS HANDLED

fig, ax = plt.subplots(figsize=(15, 15))
sns.boxplot(data=data, ax=ax)

#  TRAIN–TEST SPLIT

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

#  SCALING

scaler = StandardScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(X_test)

#  MODEL TRAINING

LR = LinearRegression()
LR.fit(x_train_norm, y_train)

#  PREDICTION
x_pred = LR.predict(x_test_norm)
print(x_pred)

#  SAVE MODEL

pickle.dump(LR, open("House_pred.pkl", "wb"))

print("Model saved as House_pred.pkl")
