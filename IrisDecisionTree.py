import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

irisData = pd.read_csv("/Users/niharika/Desktop/iris_data_set/iris.csv")

#print(irisData.head())
#print(irisData.decribe())#describes the mean, min,max ,variance etc
#print(irisData.corr())#descibes the correlation matrix

features = irisData(["SepalLength","SepalWidth","PetalLength","PetalWidth","Class"])
targetVariables = irisData.Class

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size =.2 )

model = DecisionTreeClassifier()
fittedModel = model.fit(featureTrain, targetTrain) 

predictions = fittedModel.predict(featureTest)

print(confusion_matrix(targetTest,predictions))
print(accuracy_score(targetTest,predictions))