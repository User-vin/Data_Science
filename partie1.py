import xgboost as xgb
from google.colab import drive
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
drive.mount('/content/gdrive', force_remount=True)



def add_features(X):

  h, w = X.shape
  average = []
  maxi = []
  mini = []
  median = []
  stand = []
  var = []
  abs_med = []


  for i in range(h):
    average.append(np.mean(X[i,:]))
    maxi.append(np.max(X[i,:]))
    mini.append(np.min(X[i,:]))
    median.append(np.median(X[i,:]))

    abs_deviations = np.abs(X[i,:] - np.median(X[i,:]))
    abs_med.append(np.median(abs_deviations))

    stand.append(np.std(X[i,:]))
    var.append(np.var(X[i,:]))

  average = np.array(average)
  maxi = np.array(maxi)
  mini = np.array(mini)
  median = np.array(median)
  stand = np.array(stand)
  var = np.array(var)
  abs_med = np.array(abs_med)

  new_array = np.column_stack((average, maxi))
  new_array = np.column_stack((new_array, mini))
  new_array = np.column_stack((new_array, median))
  new_array = np.column_stack((new_array, stand))
  new_array = np.column_stack((new_array, var))
  new_array = np.column_stack((new_array, abs_med))
  
  return new_array


test_data = pd.read_csv('/content/gdrive/MyDrive/InputTest.csv')
train_data = pd.read_csv('/content/gdrive/MyDrive/InputTrain.csv')
train_labels = pd.read_csv('/content/gdrive/MyDrive/StepOne_LabelTrain.csv')

test_data = test_data.drop('House_id', axis=1).drop('Index', axis=1).astype(float)
train_data = train_data.drop('House_id', axis=1).drop('Index', axis=1).astype(float)
train_labels = train_labels.drop('House_id', axis=1).drop('Index', axis=1)

X_train = train_data.to_numpy()
X_test = test_data.to_numpy()
y_train = train_labels.to_numpy()


X_train = add_features(X_train)
X_test = add_features(X_test)


def training(X_train, X_test, y_train):
  model = xgb.XGBClassifier()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  return y_pred

y_pred = training(X_train, X_test, y_train)

cols = ['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']
df = pd.DataFrame(y_pred, columns=cols)
df.index.name = 'Index'
df.to_csv('res.csv', index=True)
