import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Sample training data
data = {
    "rainfall": [
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,
        110, 120, 130, 140, 150
    ],

    "temperature": [
        32, 31, 30, 29, 29,
        28, 28, 27, 27, 26,
        26, 25, 25, 24, 24
    ],

    "humidity": [
        40, 45, 50, 55, 60,
        65, 70, 72, 75, 78,
        80, 82, 85, 88, 90
    ],

    "wind_speed": [
        8, 10, 11, 12, 13,
        14, 15, 16, 17, 18,
        19, 20, 21, 22, 23
    ],

    "water_level": [
        1, 2, 2, 3, 3,
        4, 4, 5, 5, 6,
        6, 7, 7, 8, 8
    ],

    "flood": [
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}


df = pd.DataFrame(data)

# Features
X = df[
    [
        "rainfall",
        "temperature",
        "humidity",
        "wind_speed",
        "water_level"
    ]
]

# Target
y = df["flood"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Save model
with open("flood_model.pkl", "wb") as file:
    pickle.dump(model, file)


print("Model trained successfully!")
print("Model saved as flood_model.pkl")