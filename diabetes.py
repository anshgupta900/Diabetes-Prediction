import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Diabetes_Prediction_Dataset.csv")

df.head()

df.shape

df.info()

df.describe()

df.isnull().sum()

df.drop_duplicates(inplace=True)

df["Glucose"] = df["Glucose"].fillna(df["Glucose"].mean())

df["BMI"] = df["BMI"].fillna(df["BMI"].mean())

df.isnull().sum()

df.drop('Patient_ID', axis=1, inplace=True)

df["Gender"].unique()

df['Gender'].value_counts(dropna=False)

df['Gender'] = df['Gender'].replace({
    'Unknown': np.nan
})

df['Gender'].value_counts(dropna=False)

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

df['Gender'].value_counts(dropna=False)

numerical_cols=df.columns

plt.figure(figsize=(15,8))

for i, col in enumerate(numerical_cols, 1):

    plt.subplot(4,4,i)

    sns.boxplot(y=df[col])

plt.tight_layout()

plt.show()

df['Blood_Pressure'].describe()

df['Blood_Pressure'].describe()

Q1 = df['Blood_Pressure'].quantile(0.25)
Q3 = df['Blood_Pressure'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

df['Blood_Pressure'] = np.where(
    df['Blood_Pressure'] > upper_bound,
    upper_bound,
    np.where(
        df['Blood_Pressure'] < lower_bound,
        lower_bound,
        df['Blood_Pressure']
    )
)

df['Blood_Pressure'].describe()

plt.figure(figsize=(15,8))

for i, col in enumerate(numerical_cols, 1):

    plt.subplot(4,4,i)

    sns.boxplot(y=df[col])

plt.tight_layout()

plt.show()

df.head()

df["Diabetes"] = df["Diabetes"].map({
    "No": 0,
    "Yes": 1,
    0: 0,
    1: 1
})
df['Diabetes'].isna().sum()

from sklearn.preprocessing import OneHotEncoder

categorical_cols = [
    'Gender',
    'Smoking_Status',
    'Physical_Activity',
    'Family_History'
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

log = LogisticRegression(max_iter=1000)

dt = DecisionTreeClassifier(
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

gbr = GradientBoostingClassifier(
    random_state=42
)

xg = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss'
)

svm = SVC()

log.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)
xg.fit(X_train, y_train)
svm.fit(X_train, y_train)

y_pred1 = log.predict(X_test)
y_pred2 = dt.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gbr.predict(X_test)
y_pred5 = xg.predict(X_test)
y_pred6 = svm.predict(X_test)

from sklearn.metrics import accuracy_score

models = {
    'Logistic Regression': log,
    'Decision Tree': dt,
    'Random Forest': rf,
    'Gradient Boosting': gbr,
    'XGBoost': xg,
    'SVM': svm
}

results = {}

for name, model in models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(
        f"{name}: "
        f"{accuracy * 100:.2f}%"
    )

best_model_name = max(results, key=results.get)

# Get the actual trained model
best_model = models[best_model_name]

# Get accuracy
best_accuracy = results[best_model_name]

print("Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy * 100, "%")


# Save the actual trained model
import joblib

joblib.dump(
    best_model,
    'best_diabetes_model.pkl'
)

print("Best model saved successfully!")
model = joblib.load('best_diabetes_model.pkl')

print(type(model))