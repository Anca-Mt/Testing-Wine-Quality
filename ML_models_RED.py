import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

df_red = pd.read_csv('Wine DATASET/RED/winequality-red.csv')
#df_red = df_red.drop(labels= ['free sulfur dioxide', 'pH', 'residual sugar', 'fixed acidity', 'citric acid'], axis=1)
#df = df_red.drop(labels=["citric acid", "pH", "density"], axis=1)

# Create the y array
# y_series = df_red.iloc[:,11]
# y = pd.Series.to_numpy(y_series)
# or: y=np.array(df_red.iloc[:,11])
y_series = df_red.iloc[:, -1]
converter = LabelEncoder()
y=converter.fit_transform(y_series)

# SCALING THE FEATURES: standard scaler

scaler = StandardScaler()
X = scaler.fit_transform(np.array(df_red.iloc[:, :-1], dtype=float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=40)

#
# # DECISION TREE
# tree_clf = DecisionTreeClassifier(max_depth=None)
# tree_clf.fit(X_train, y_train)
# y_test_pred = tree_clf.predict(X_test)


# Random Forest
rnd_clf = RandomForestClassifier(n_estimators=200)
# rnd_clf.fit(X_train,y_train)
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

# #### Extract features importances
# print('FEATURE IMPORTANCE:')
# print(rnd_clf.feature_importances_)
# feat_importances = pd.Series(rnd_clf.feature_importances_, index=df_red.iloc[:, :11].columns)
# feat_importances.nlargest(11).plot(kind='barh')
# plt.show()

# => deciding to drop the following columns:
# free sulfur dioxide; pH; residual sugar; fixed acidity; citric acid

# ## POLYNOMIAL KERNEL SVC
# svm_clf = Pipeline([("svm_clf", SVC(kernel='poly', C=10, degree=6))])
# svm_clf.fit(X_train, y_train)
# y_test_pred = svm_clf.predict(X_test)


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

