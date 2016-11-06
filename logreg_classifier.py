from scipy import misc
from sklearn import linear_model as lm
import os
import numpy as np
import csv

# Read the data
categories = ["airplane", "car", "castle", "church", "coffee mug",
              "dog", "fruit", "house", "table", "teapot", "volcano"]
directory = os.curdir

def readTestData(test_dir):
    test = []
    true_labels = []
    test_filename = []
    for filename in os.listdir(directory + test_dir):
        category_found = False
        for category in categories:
            if not category_found and filename.find(category) >= 0:
                image = misc.imread(directory + test_dir + "\\" + filename)
                image = misc.imresize(image, (224, 224, 3))
                test.append(image.flatten())
                test_filename.append(filename)
                true_labels.append(category)
                category_found = True  # Just to make sure there is only one category found.
    return test, true_labels, test_filename

def writeResults(filename, model, test, true_labels, test_filename):
    prob_results = model.predict_proba(test)
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in range(len(test)):
            label = true_labels[i]
            col = np.where(model.classes_ == label)
            prob = prob_results[i][col][0]
            writer.writerow([test_filename[i], prob])


print('Reading Train data')
X = []
y = []
i = 0
for category in categories:
    for filename in os.listdir(directory + "\\data\\" + category):
        if True:  #i < 20:
            image = misc.imread(directory + "\\data\\" + category + "\\" + filename)
            image = misc.imresize(image, (224, 224, 3))
            if image.shape == (224, 224, 3):
                X.append(image.flatten())
                y.append(category)
            else:
                print(filename)
                i += 1

print(i)
# print(set([image.shape for image in X]))

# Test data
print('Reading test data')
test1, true_labels1, test_filename1 = readTestData('\\Analysis\\pictures1')
test2, true_labels2, test_filename2 = readTestData('\\Analysis\\pictures2')
test3, true_labels3, test_filename3 = readTestData('\\Analysis\\pictures3')

# Logistic Regression
print('Training Classifier')
model = lm.LogisticRegression()
model.fit(X, y)

# Results
print('Writing Results')
writeResults('logreg_results\\all1.csv', model, test1, true_labels1, test_filename1)
writeResults('logreg_results\\all2.csv', model, test2, true_labels2, test_filename2)
writeResults('logreg_results\\all3.csv', model, test3, true_labels3, test_filename3)
