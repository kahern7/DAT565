import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

# Load in Iris data
from sklearn.datasets import load_iris
iris = load_iris()
class_names = iris.target_names

# Split data into training and datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

#############################
# Logistic Regression
#############################

# Modelling Data
logisticRegr = LogisticRegression(multi_class='ovr', solver='liblinear')
logisticRegr.fit(x_train, y_train)
logisticRegr.predict(x_test[0].reshape(1,-1))
logisticRegr.predict(x_test[0:10])
predictions_lr = logisticRegr.predict(x_test)

# Measure accuracy of data using score method
score = logisticRegr.score(x_test, y_test)
print(score)

cm = metrics.confusion_matrix(y_test, predictions_lr)

# Plot non-normalized confusion matrix
title = "Confusion matrix, without normalization"
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format="d")
disp.ax_.set_title(title)
plt.savefig('toy_iris_ConfusionSeabornCodementor.png')

#############################
# K Nearest Neighbours
#############################

weight = ['uniform', 'distance']
n_value = [1, 10, 25, 50, 100]

# Create for loop for different k and n
for weights in weight:
    for n_neighbours in n_value:

        # create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbours, weights=weights) #k=1,3,21

        # train the model using the training sets
        knn.fit (x_train, y_train)

        # predict the response for dataset
        y_pred_knn = knn.predict(x_test)

        # model accuracy, how often is the classifier correct?
        print("Accuracy for k={} and weights={}:".format(n_neighbours, weights), metrics.accuracy_score(y_test, y_pred_knn))

        cm_knn = metrics.confusion_matrix(y_test, y_pred_knn)

        # Plot non-normalized confusion matrix
        title = "Confusion matrix for {} weight & k={}, without normalization".format(weights, n_neighbours)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        disp.ax_.set_title(title)
        #plt.savefig('iris_plot_k_vlaues')

knn.predict(iris.data[10:11])
iris.target[10:11]

# Save plot and show
plt.show()