import sys
# print('Python: {}' .format(sys.version))
import scipy
# print('scipy: {}' .format(scipy.__version__))
import numpy
# print('numpy: {}' .format(numpy.__version__))
import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
# print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))



from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#import classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#Dimensions of the dataset
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())

#find distribution
print(dataset.groupby('class').size())

#Visualizations
print("Visualizations")
dataset.plot(kind='box',subplots = True, layout = (2,2), sharex  =  False,sharey = False)
#plt.show()

dataset.hist()
#plt.show()


# scatter plot matrix
scatter_matrix(dataset)
plt.show()

####Evaluating + training algorithms

#validation dataset
array = dataset.values
X = array[:,0:4] #sepal length, width, petal length, width
Y = array[:,4] #class

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#setting up training harness
scoring = 'accuracy' #correctly predicted instances/ total instancces

#building models
models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

#evaluating
results = []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold , scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,  cv_results.mean(), cv_results.std())
    print(msg)


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#testing on validation, results show that KNN was the most accuarte model  (0.983)
knn  =  KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))