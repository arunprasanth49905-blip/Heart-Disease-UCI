import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# Load dataset

data = pd.read_csv("heart.csv")


# 13 features used by UCI dataset

features = [

"age",
"sex",
"cp",
"trestbps",
"chol",
"fbs",
"restecg",
"thalach",
"exang",
"oldpeak",
"slope",
"ca",
"thal"

]


X = data[features]

y = data["target"]



# Split dataset

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)



# Model

model = Pipeline([

    ("scaler", StandardScaler()),

    ("classifier",
     RandomForestClassifier(
         n_estimators=150,
         random_state=42
     ))

])



# Train

model.fit(
    X_train,
    y_train
)



# Accuracy

pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    pred
)


print(
    "Accuracy:",
    accuracy*100
)



# Save model

pickle.dump(

    model,

    open(
        "model.pkl",
        "wb"
    )

)


print("model.pkl created")
