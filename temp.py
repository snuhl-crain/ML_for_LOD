# 24.11.12 edited '''maybe''' final source code

#!pip install pycaret
#!pip install markupsafe==2.0.1
#!pip install shap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from pycaret.classification import *

# train/test data generator

file_path = ''
testing = ''
trained = ''
data = pd.read_csv(file_path+'.csv')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=999)
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv(file_path+'train.csv', index=True)
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv(file_path+'test.csv', index=True)

# model trainer

train_data = train_data
test_data = test_data

s = setup(data=train_data, test_data=test_data, target='True Class', session_id=999, use_gpu=True, categorical_features = ['Gender','Low income','Big city','Living alone','Dentures','Pain','Social meeting'], fix_imbalance=True,index=False)

models = []
using_models=['lightgbm','xgboost','rf','gbc','et','ada','lda','lr','nb','knn','qda','dt']
class_names = ['0', '1']
for i in using_models:
    print(i)
    globals()['model_{}'.format(str(i))] = create_model(i)
    globals()['tuned_{}'.format(str(i))] = tune_model(globals()['model_{}'.format(str(i))],optimize='mcc', n_iter=50)
    pred= predict_model(globals()['tuned_{}'.format(str(i))], data=test_data,raw_score=True)
    pred.to_csv(i+'.csv', index=False)
    models.append(globals()['tuned_{}'.format(str(i))])
    globals()['final_{}'.format(str(i))] = finalize_model(globals()['tuned_{}'.format(str(i))])
    save_model(globals()['tuned_{}'.format(str(i))], file_path+i+testing+trained+'_tuned_model')
    save_model(globals()['final_{}'.format(str(i))], file_path+i+testing+trained+'_final_model')
    cm = confusion_matrix(pred['True Class'], pred['prediction_label'])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=class_names, yticklabels=class_names, vmax=500, vmin=0)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# calculate feature importance/coefficient

feat_imp_dict = {}
models = []

using_models=['lightgbm','xgboost','rf','gbc','et','ada','dt']
for i in using_models:
    print(i)
    plot_model(globals()['tuned_{}'.format(str(i))], plot = 'feature_all')
    feat_importance = globals()['tuned_{}'.format(str(i))].feature_importances_
    feat_imp_dict[i] = feat_importance.flatten()
  
using_models=['lr','lda']
for i in using_models:
    print(i)
    plot_model(globals()['tuned_{}'.format(str(i))], plot = 'feature_all')
    feat_importance = globals()['tuned_{}'.format(str(i))].coef_
    feat_imp_dict[i] = feat_importance.flatten()
feat_imp_df = pd.DataFrame(feat_imp_dict)

# caculate shap value

import shap

shap_models=['gbc']
shap.initjs()
for i in shap_models:
    print(i)
    explainer = shap.TreeExplainer(globals()['tuned_{}'.format(str(i))].named_steps['trained_model'])
    shap_values = explainer.shap_values(train_data.iloc[:, :-1])
    shap.summary_plot(shap_values, train_data.iloc[:, :-1])

shap_models=['rf']
shap.initjs()
for i in shap_models:
    print(i)
    explainer = shap.TreeExplainer(globals()['tuned_{}'.format(str(i))].named_steps['trained_model'])
    shap_values = explainer.shap_values(train_data.iloc[:, :-1])
    shap_values = shap_values[:, :, 1]
    shap.summary_plot(shap_values, train_data.iloc[:, :-1])

# drawing ROC curve

using_models=['lightgbm','gbc','rf', 'et','lr','lda','qda','dt','xgboost','ada','nb','knn']
Using_models=['LIGHTGBM','GBC','RF','ET','LR','LDA','QDA','DT','XGBOOST','ADA','NB','KNN']

file_paths = [f'{i}.csv' for i in using_models]
data_list = [pd.read_csv(file_path) for file_path in file_paths]

# Extract the relevant columns and calculate AUC scores and ROC curve data
auc_scores = []
fpr_list = []
tpr_list = []

for i, data in enumerate(data_list):
    print(i)
    y_true = data['True Class']
    y_pred = data['prediction_score_1']
    y_true_binary = (y_true == 1).astype(int)
    auc = roc_auc_score(y_true_binary, y_pred)
    auc_scores.append(auc)
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
    fpr_list.append(fpr)
    tpr_list.append(tpr)

plt.figure(figsize=(14, 10))
j = 0
for i in Using_models:
    plt.plot(fpr_list[j], tpr_list[j], label=f'{i} (AUC = {auc_scores[j]:.2f})', linewidth=2)
    j=j+1

plt.plot([0, 1], [0, 1], 'k--', label='Random chance', linewidth=2)
plt.xlabel('1 - Specificity', fontsize=30)
plt.ylabel('Sensitivity', fontsize=30)
plt.legend(loc='best', fontsize=18)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=20)
plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=20)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300)

# Evaluate models

results = []

def calculate_metrics(true_labels, predictions):
    accuracy = round(accuracy_score(true_labels, predictions),2)
    precision_group1 = round(precision_score(true_labels, predictions, pos_label=1),2)
    recall_group1 = round(recall_score(true_labels, predictions, pos_label=1),2)
    f1_group1 = round(f1_score(true_labels, predictions, pos_label=1),2)
    mcc = round(matthews_corrcoef(true_labels, predictions),2)
    return [accuracy, precision_group1, recall_group1, f1_group1, mcc]

for file_path in file_paths:
    # Read the CSV file
    df = pd.read_csv(file_path)  # Adjust encoding if necessary
    
    if 'true_label' in df.columns and 'prediction_label' in df.columns:
        true_labels = df['true_label']
        predictions = df['prediction_label']
        metrics = calculate_metrics(true_labels, predictions)
        results.append([file_path] + metrics)

columns = ['file_path', 'accuracy', 'precision_group1', 'recall_group1', 'f1_group1', 'mcc']
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv('classification_metrics_results.csv', index=False)



