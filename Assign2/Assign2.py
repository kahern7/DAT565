import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Load in Iris data
from sklearn.datasets import load_iris
iris = load_iris()
class_names = iris.target_names

# Split data into training and datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

# Print Images
print("Image Data Shape", iris.data.shape)

# Print labels
print("Label Data Shape", iris.target.shape)

# # Create plot of data
# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(iris.data[0:5], iris.target[0:5])):
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(image, (2,2)), cmap=plt.cm.gray)
#     plt.title('Training: %i\n' % label, fontsize = 20)

# Modelling Data
logisticRegr = LogisticRegression(multi_class='ovr', solver='liblinear')
logisticRegr.fit(x_train, y_train)
logisticRegr.predict(x_test[0].reshape(1,-1))
logisticRegr.predict(x_test[0:10])
predictions = logisticRegr.predict(x_test)

# Measure accuracy of data using score method
score = logisticRegr.score(x_test, y_test)
print(score)

cm = metrics.confusion_matrix(y_test, predictions)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Confusion matrix, without nomralisation'
plt.title(all_sample_title, size = 15)
plt.savefig('toy_iris_ConfusionSeabornCodementor.png')

plt.show()