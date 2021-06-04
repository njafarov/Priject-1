# Project-1 MNIST
Alice, Bob, and Daniel are friends learning machine learning together. After watching a few lectures, they are very proud of having learned many useful tools, including linear and logistic regression, non-linear features, regularization, and kernel tricks. To see how these methods can be used to solve a real life problem, they decide to get their hands dirty with the famous digit recognition problem using the MNIST (Mixed National Institute of Standards and Technology) database.

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project, you will get a chance to experiment with the task of classifying these images into the correct digit using some of the methods you have learned so far.


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

"""Cross Validation for MNIST data"""

y_train_5 = (y_train==5) #selects 5 from train data
y_test_5 = (y_test==5) #selects 5 from test data

sdg_clf = SGDClassifier(random_state=42) # SGD classifier that classifes hand written digits
sgd_clf.fit(X_train, y_train_5)


str_kfol = StratifiedKFold(n_splits=3)

for train_index, test_index in str_kfol.split(X_train, y_train_5): #split stratified kfold data and also iterates
    clone_clf = clone(sgd_clf) #with the help of clone, we clone classifier with all its parameters
    X_train_fold = X_train[train_index] #We selec a some data for training
    X_test_fold = X_train[test_index] #data selection for X_test
    y_train_fold = y_train_5[train_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_fold, y_train_fold) #We can run the cloned clussifier with our splitted data
    y_predict = clone_clf.predict(X_test_fold)
    n_correct = sum(y_predict==y_test_fold) #We take the sum of all correctly clasified data
    print(n_correct/len(y_predict))
