#  Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load Dataset Fresh

df = pd.read_excel("/Users/raghdalzulafi/Desktop/RAHHAL/universal_data.xlsx")
print("Original Shape:", df.shape)
print(df.head())

# Data Cleaning & Preprocessing

df = df.drop_duplicates()

df["diseases"] = df["diseases"].fillna("none")
df["Fear_of_Heights"] = df["Fear_of_Heights"].fillna("no")

text_columns = [
    "Age_Group",
    "diseases",
    "accompanied_with",
    "preference",
    "best_fit",
    "examples",
    "Fear_of_Heights"
]

for col in text_columns:
    df[col] = df[col].astype(str).str.lower().str.strip()

df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
df["Height_cm"] = pd.to_numeric(df["Height_cm"], errors="coerce")

df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
df["Height_cm"] = df["Height_cm"].fillna(df["Height_cm"].median())

df = df[(df["weight_kg"] > 30) & (df["weight_kg"] < 200)]
df = df[(df["Height_cm"] > 120) & (df["Height_cm"] < 220)]

print("After Cleaning:", df.shape)


# Final Validation
print("\nMissing Values:\n", df.isnull().sum())
print("\nSample Data:\n", df.head())


#  Smart Simplification (Correct Way)

# ناخذ أول جملة قبل أول نقطة فقط
df["best_fit"] = df["best_fit"].apply(lambda x: x.split(".")[0].strip())

print("\nNew best_fit classes:", df["best_fit"].nunique())
print(df["best_fit"].value_counts())

#  Define Features & Targets

X = df[[
    "Age_Group",
    "weight_kg",
    "diseases",
    "accompanied_with",
    "preference",
    "Height_cm",
    "Fear_of_Heights"
]].copy()

y1 = df["best_fit"].copy()
y2 = df["examples"].copy()

#  Encoding

encoders_X = {}

for col in X.columns:
    if col in text_columns:
        X[col] = X[col].astype(str)

# 🔥 ثم نسوي encoding لها
for col in X.columns:
    if col in text_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders_X[col] = le
le_y1 = LabelEncoder()
le_y2 = LabelEncoder()

y1 = le_y1.fit_transform(y1)
y2 = le_y2.fit_transform(y2)

num_classes_y1 = len(np.unique(y1))
num_classes_y2 = len(np.unique(y2))

y1 = keras.utils.to_categorical(y1, num_classes_y1)
y2 = keras.utils.to_categorical(y2, num_classes_y2)


X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y1, y2,
    test_size=0.2,
    random_state=42
)


input_layer = layers.Input(shape=(X.shape[1],))

shared = layers.Dense(256, activation='relu')(input_layer)
shared = layers.Dropout(0.2)(shared)
shared = layers.Dense(128, activation='relu')(shared)
shared = layers.Dropout(0.2)(shared)
shared = layers.Dense(64, activation='relu')(shared)

output1 = layers.Dense(num_classes_y1, activation='softmax', name="best_fit")(shared)
output2 = layers.Dense(num_classes_y2, activation='softmax', name="examples")(shared)

model = keras.Model(inputs=input_layer, outputs=[output1, output2])

model.compile(
    optimizer='adam',
    loss={
        "best_fit": "categorical_crossentropy",
        "examples": "categorical_crossentropy"
    },
    metrics={
        "best_fit": "accuracy",
        "examples": "accuracy"
    }
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    [y1_train, y2_train],
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


results = model.evaluate(
    X_test,
    [y1_test, y2_test],
    verbose=0
)

print("\nBest Fit Accuracy: {:.2f}%".format(results[3] * 100))
print("Examples Accuracy: {:.2f}%".format(results[4] * 100))


# Predictions
pred_y1, pred_y2 = model.predict(X_test)

pred_y1_classes = np.argmax(pred_y1, axis=1)
pred_y2_classes = np.argmax(pred_y2, axis=1)

true_y1_classes = np.argmax(y1_test, axis=1)
true_y2_classes = np.argmax(y2_test, axis=1)

# Individual Accuracies
best_fit_acc = np.mean(pred_y1_classes == true_y1_classes)
examples_acc = np.mean(pred_y2_classes == true_y2_classes)

# Joint Accuracy (both correct)
joint_acc = np.mean(
    (pred_y1_classes == true_y1_classes) &
    (pred_y2_classes == true_y2_classes)
)

# Average Accuracy
overall_acc = (best_fit_acc + examples_acc) / 2

print(f"\nBest Fit Accuracy: {best_fit_acc*100:.2f}%")
print(f"Examples Accuracy: {examples_acc*100:.2f}%")
print(f"Joint Accuracy (Both Correct): {joint_acc*100:.2f}%")
print(f"Overall Model Accuracy: {overall_acc*100:.2f}%")

plt.figure(figsize=(12,5))

# Best Fit Accuracy Curve
plt.subplot(1,2,1)
plt.plot(history.history['best_fit_accuracy'], label="Train")
plt.plot(history.history['val_best_fit_accuracy'], label="Validation")
plt.title("Best Fit Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Examples Accuracy Curve
plt.subplot(1,2,2)
plt.plot(history.history['examples_accuracy'], label="Train")
plt.plot(history.history['val_examples_accuracy'], label="Validation")
plt.title("Examples Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

def predict_user():

    print("\n=== User Input ===\n")

    age_options = sorted(df["Age_Group"].unique())
    disease_options = sorted(df["diseases"].unique())
    group_options = sorted(df["accompanied_with"].unique())
    pref_options = sorted(df["preference"].unique())
    fear_options = sorted(df["Fear_of_Heights"].unique())

    # عرض الخيارات
    print("Age Group options:", age_options)
    age = input("Choose Age Group: ").lower().strip()

    weight = float(input("Enter Weight (kg): "))

    print("Diseases options:", disease_options)
    disease = input("Choose Disease: ").lower().strip()

    print("Group options:", group_options)
    group = input("Choose Group Type: ").lower().strip()

    print("Preference options:", pref_options)
    preference = input("Choose Preference: ").lower().strip()

    height = float(input("Enter Height (cm): "))

    print("Fear options:", fear_options)
    fear = input("Fear of Heights: ").lower().strip()

    input_data = pd.DataFrame([{
        "Age_Group": age,
        "weight_kg": weight,
        "diseases": disease,
        "accompanied_with": group,
        "preference": preference,
        "Height_cm": height,
        "Fear_of_Heights": fear
    }])

    # Encoding
    for col in input_data.columns:
        if col in encoders_X:
            input_data[col] = encoders_X[col].transform(input_data[col])

    # Prediction
    pred1, pred2 = model.predict(input_data)

    best_fit_pred = le_y1.inverse_transform([np.argmax(pred1)])
    examples_pred = le_y2.inverse_transform([np.argmax(pred2)])

    print("\n Best Fit Recommendation:", best_fit_pred[0])
    print(" Example Activities:", examples_pred[0])


# Save Model & Encoders

model.save("models/API/recommendation_model.h5")

import pickle

with open("models/API/encoders_X.pkl", "wb") as f:
    pickle.dump(encoders_X, f)

with open("models/API/label_encoder_y1.pkl", "wb") as f:
    pickle.dump(le_y1, f)

with open("models/API/label_encoder_y2.pkl", "wb") as f:
    pickle.dump(le_y2, f)