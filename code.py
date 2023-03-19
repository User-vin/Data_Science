import xgboost as xgb
from google.colab import drive
import numpy as np
import pandas as pd

#drive.mount('/content/gdrive', force_remount=True)


train_labels = pd.read_csv('/content/gdrive/MyDrive/StepOne_LabelTrain.csv')
train_data = pd.read_csv('/content/gdrive/MyDrive/InputTrain.csv')

train_data = train_data.drop('House_id', axis=1).drop('Index', axis=1).astype(float)
train_labels = train_labels.drop('House_id', axis=1).drop('Index', axis=1).astype(float)

X = train_data.to_numpy()
y = train_labels.to_numpy()

# define the model
model = xgb.XGBClassifier()

# fit the model on the training data
model.fit(X, y)

test_data = pd.read_csv('/content/gdrive/MyDrive/InputTest.csv')
test_data = test_data.drop('House_id', axis=1).drop('Index', axis=1).astype(float)
X = test_data.to_numpy()
arr = model.predict(X)
arr = np.round(arr).astype(np.uint8)
cols = ['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']
df = pd.DataFrame(arr, columns=cols)
df.index.name = 'Index'
df.to_csv('ress.csv', index=True)
