from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('Wine DATASET/WHITE/winequality-white.csv')
# df = df.drop(labels=["citric acid", "pH", "density", "total sulfur dioxide"], axis=1)

y_series = df.iloc[:, -1]
converter = LabelEncoder()
y=converter.fit_transform(y_series)

X = np.array(df.iloc[:, :-1], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=898, random_state=40)


# ### FISHER'S SCORE
# # Calculating scores
# ranks = fisher_score.fisher_score(X_train, y_train)
#
# # Plotting the ranks
# feat_importances = pd.Series(ranks, df.columns[0:11])
# feat_importances.plot(kind='barh', color='teal')
# plt.show()

# ### VARIANCE THRESHOLD
# v_threshold = VarianceThreshold(threshold=0)
# v_threshold.fit(X_train)
# v_threshold.get_support()

### Correlation Coefficient
cor = df.iloc[:, :-1].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(cor, annot=True)
plt.show()