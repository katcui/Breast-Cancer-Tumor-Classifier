# Load libraries
# from pandas import read_csv

import pandas as pd 
import seaborn as sns
import pickle

from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
target = ['diagnosis',	'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
df = pd.read_csv("data.csv")

target_columns = df[target]

sns.pairplot(target_columns, hue='diagnosis')
plt.show()

# Dimensions of Dataset
# shape
print(target_columns.shape) # (569, 5)

# head
print(target_columns.head(20)) # first 20 rows of data

# Statistical Summary
print(target_columns.describe()) 

# class distribution
print(target_columns.groupby('diagnosis').size()) # 357 benign, 212 malignant

# Data Visualization

# box and whisker plots
target_columns.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
target_columns.hist()
plt.show()
# scatter plot matrix
scatter_matrix(target_columns)
plt.show()

# Split-out validation dataset
array = target_columns.values
X = array[:,1:5] # input values
y = array[:,0] # output values

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# Make predictions on validation dataset
LR_model = LogisticRegression()
LR_model.fit(X_train, Y_train)
LR_predictions = LR_model.predict(X_validation)

LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(X_train, Y_train)
LDA_predictions = LDA_model.predict(X_validation)

pickle.dump(LR_model, open('LR_model.pkl','wb'))
pickle.dump(LDA_model, open('LDA_model.pkl','wb'))

# Evaluate predictions
print(accuracy_score(Y_validation, LR_predictions))
print(confusion_matrix(Y_validation, LR_predictions))
print(classification_report(Y_validation, LR_predictions))

print(accuracy_score(Y_validation, LDA_predictions))
print(confusion_matrix(Y_validation, LDA_predictions))
print(classification_report(Y_validation, LDA_predictions))