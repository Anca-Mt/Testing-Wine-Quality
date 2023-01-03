import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df_red = pd.read_csv('Wine DATASET/RED/winequality-red.csv')
df = df_red.drop(labels=["citric acid", "pH", "density"], axis=1)

y_series = df_red.iloc[:, -1]
converter = LabelEncoder()
y=converter.fit_transform(y_series)

# SCALING THE FEATURES: standard scaler

scaler = StandardScaler()
X = scaler.fit_transform(np.array(df_red.iloc[:, :-1], dtype=float))
# X = np.array(df_red.iloc[:, :-1], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=40)

logr = LogisticRegression()
logr.fit(X_train, y_train)

predictions = logr.predict_proba(X_test)
print('Predictions: ', np.around(predictions,2))
y_pred = logr.predict(X_test)

print("Accuracy: ", accuracy_score(y_true=y_test,y_pred=y_pred))
print(classification_report(y_test,y_pred))
