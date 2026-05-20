import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("ai4i2020.csv")

# =========================================================
# CREATE TARGET COLUMN
# =========================================================

def get_failure_type(row):
    if row['TWF'] == 1:
        return 'TWF'
    elif row['HDF'] == 1:
        return 'HDF'
    elif row['PWF'] == 1:
        return 'PWF'
    elif row['OSF'] == 1:
        return 'OSF'
    elif row['RNF'] == 1:
        return 'RNF'
    else:
        return 'No Failure'

df['fault_type'] = df.apply(get_failure_type, axis=1)

# =========================================================
# PREPARE FEATURES
# =========================================================

drop_cols = [
    'UDI',
    'Product ID',
    'Machine failure',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
]

X = df.drop(columns=drop_cols)
y = df['fault_type']

# Remove target from features
X = X.drop(columns=['fault_type'])

# Encode Type column
le = LabelEncoder()
X['Type'] = le.fit_transform(X['Type'])

# =========================================================
# FEATURE ENGINEERING
# =========================================================

X['temp_difference'] = (
    X['Process temperature [K]']
    - X['Air temperature [K]']
)

X['power'] = (
    X['Rotational speed [rpm]']
    * X['Torque [Nm]']
    * 2
    * 3.14159
    / 60
)
# =========================================================
# FEATURE ENGINEERING
# =========================================================

X['temp_difference'] = (
    X['Process temperature [K]']
    - X['Air temperature [K]']
)

X['power'] = (
    X['Rotational speed [rpm]']
    * X['Torque [Nm]']
    * 2
    * 3.14159
    / 60
)

X['wear_torque'] = (
    X['Tool wear [min]']
    * X['Torque [Nm]']
)

X['temp_ratio'] = (
    X['Process temperature [K]']
    / X['Air temperature [K]']
)
# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# SMOTE
# =========================================================

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

# =========================================================
# TRAIN MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_resampled, y_train_resampled)

# =========================================================
# CROSS VALIDATION
# =========================================================

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='f1_macro'
)

print("\nCROSS VALIDATION SCORES:")
print(scores)

print("\nMEAN CV MACRO F1:")
print(scores.mean())

# =========================================================
# PREDICT
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# EVALUATION
# =========================================================

macro_f1 = f1_score(
    y_test,
    y_pred,
    average='macro'
)

print("\nMACRO F1 SCORE:")
print(macro_f1)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# =========================================================
# CONFUSION MATRIX
# =========================================================

labels = [
    "No Failure",
    "HDF",
    "OSF",
    "PWF",
    "TWF",
    "RNF"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=150)

plt.show()