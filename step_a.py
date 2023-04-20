import xgboost as xgb
from google.colab import drive
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
drive.mount('/content/gdrive', force_remount=True)


#Function that takes 2d array and returns an ndarray with 7 columns (the features extracted from the data)
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

#Read CSVs
test_data = pd.read_csv('/content/gdrive/MyDrive/InputTest.csv')
train_data = pd.read_csv('/content/gdrive/MyDrive/InputTrain.csv')
train_labels = pd.read_csv('/content/gdrive/MyDrive/StepOne_LabelTrain.csv')

#Remove House_id and Index columns
test_data = test_data.drop('House_id', axis=1).drop('Index', axis=1).astype(float)
train_data = train_data.drop('House_id', axis=1).drop('Index', axis=1).astype(float)
train_labels = train_labels.drop('House_id', axis=1).drop('Index', axis=1)

#Convert CSVs to numpy
X_train = train_data.to_numpy()
X_test = test_data.to_numpy()
y_train = train_labels.to_numpy()

#Feature extraction
X_train = add_features(X_train)
X_test = add_features(X_test)

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

#function to train and save results plots
def training(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #Metrics for each appliances
    f1_scores = f1_score(y_test, y_pred, average=None)
    recalls = recall_score(y_test, y_pred, average=None)
    precisions = precision_score(y_test, y_pred, average=None)

    #Average metrics
    print(f1_score(y_test, y_pred, average='macro'))
    print(recall_score(y_test, y_pred, average='macro'))
    print(precision_score(y_test, y_pred, average='macro'))

    print(f1_scores)
    print(recalls)
    print(precisions)

    #Saving plots
    classes = ['Dishwasher', 'Kettle', 'Microwave', 'Tumble Dry.', 'Washing mach.']

    plt.bar(classes, f1_scores)
    plt.xlabel('Class')
    plt.ylabel('F1-score')
    plt.title('F1-scores')
    plt.savefig('f1_scores.png')
    plt.close()

    plt.bar(classes, recalls)
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title('Recalls')
    plt.savefig('recalls.png')
    plt.close()

    plt.bar(classes, precisions)
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Precisions')
    plt.savefig('precisions.png')
    plt.close()

    #Confusion matrix
    ml_cm = multilabel_confusion_matrix(y_test, y_pred)
    for i, cm in enumerate(ml_cm):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(classes[i])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()

        plt.savefig(f'confusion_matrix_class_{i + 1}.png')
        plt.close(fig)

training(X_train, X_test, y_train, y_test)
