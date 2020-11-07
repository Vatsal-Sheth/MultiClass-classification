# -*- coding: utf-8 -*-

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""_Read the dataset_"""

df = pd.read_csv('IRIS.csv')
df.head()

df.shape

df['species'].unique()

df.isnull().values.any()

df['species'].value_counts()

df['species'] = df['species'].map({'Iris-setosa' :0, 'Iris-versicolor' :1, 'Iris-virginica' :2}).astype(int)

df.tail()

df.head()

"""_Exploratory Data Analysis_"""

sns.set_style("whitegrid");
sns.FacetGrid(df, hue='species', size=5) \
    .map(plt.scatter, "sepal_length", "sepal_width") \
    .add_legend();
plt.show()

plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="species", height=3);
plt.show()

sns.cubehelix_palette()

"""EDA Inferences:
<br>1. Species 0 is much easily identifiable then 1 & 2 as they both have some overlap
<br>2. Petal length and width are the most important features.

_Normalising the dataset_
"""

x_data = df.drop(['species'],axis=1)
y_data = df['species']

MinMaxScaler = preprocessing.MinMaxScaler()
X_data_minmax = MinMaxScaler.fit_transform(x_data)

data = pd.DataFrame(X_data_minmax,columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

data.head()

"""_Implementing KNN algorithm_"""

X_train, X_test, y_train, y_test = train_test_split(data, y_data,
                                                    test_size=0.2, random_state = 1)

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_train)

ypred=knn_clf.predict(X_test)

from sklearn import metrics
metrics.accuracy_score(y_test, ypred)

"""_Visualisng Results_"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, ypred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, ypred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,ypred)
print("Accuracy:",result2)

"""**Conclusion:**
<br> We successfully implemented a KNN algorithm for the IRIS datset. We found out the most impactful deatures through out EDA and normalised our dataset for improved accuracy.  We got an accuracy of 96.67% with our algorithm as well as we got the confusion matrix and the classification report.
"""
