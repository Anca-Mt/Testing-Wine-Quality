import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

df_white = pd.read_csv('Wine DATASET/WHITE/winequality-white.csv')

y_series = df_white.iloc[:, -1]
converter = LabelEncoder()
y=converter.fit_transform(y_series)

# scaler = StandardScaler()
# X = scaler.fit_transform(np.array(df_red.iloc[:, :-1], dtype=float))
X = np.array(df_white.iloc[:, :-1], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=898, random_state=40)

print("Old train set dimensions: ", X_train.shape, y_train.shape)

# outliers identification
isf = IsolationForest(contamination=0.1)
ee = EllipticEnvelope(contamination=0.01)
lof = LocalOutlierFactor()
onesvm = OneClassSVM(nu=0.01)
y_hat = onesvm.fit_predict(X_train)

# outliers elimination
mask = y_hat != -1
X_train , y_train = X_train[mask, :], y_train[mask]

print("New train set dimensions: ", X_train.shape, y_train.shape)

rnd_clf = RandomForestClassifier(n_estimators=200)
# rnd_clf.fit(X_train, y_train)
# y_test_pred = rnd_clf.predict(X_test)

skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=10)
list_of_accuracies =[]

for train_index, valid_index in skf.split(X_train, y_train):
    X_train_fold, X_valid_fold = X[train_index], X[valid_index]
    y_train_fold, y_valid_fold = y[train_index], y[valid_index]
    rnd_clf.fit(X_train_fold, y_train_fold)
    y_valid_pred = rnd_clf.predict(X_valid_fold)
    list_of_accuracies.append(accuracy_score(y_true=y_valid_fold, y_pred=y_valid_pred))


print("List of accuracies: ", list_of_accuracies)
print("The best accuracy: ", max(list_of_accuracies))
print("Minimum accuracy: ", min(list_of_accuracies))
print("Overall accuracy: ", mean(list_of_accuracies))

y_test_pred = rnd_clf.predict(X_test)



print("Accuracy for wine quality: ", accuracy_score(y_true=y_test, y_pred=y_test_pred))
print(classification_report(y_test,y_test_pred))

matrix = confusion_matrix(y_test, y_test_pred)
print(matrix)

# displaying the confusion_matrix
display = ConfusionMatrixDisplay(matrix)

# Plotting Confusion Matrix
# Setting colour map to be used
display.plot(cmap='Greens',  xticks_rotation=25)
# Other possible options for colour map are:
# 'autumn_r', 'Blues', 'cool', 'Greens', 'Greys', 'PuRd', 'copper_r'

# Setting fontsize for xticks and yticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# Giving name to the plot
plt.title('Confusion Matrix', fontsize=24)

# Showing the plot
plt.show()
