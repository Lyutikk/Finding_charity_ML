# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 22:06:39 2022

@author: Admin
"""

import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
# import visuals as vs
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

'''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
'''


# TODO:        Func train predict
# =======================================================

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    
    results = {}
    
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    results['train_time'] = end - start
    start = time() # Get start time
    
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() 
    
    results['pred_time'] = end - start
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = 0.5)   
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)   
    
    print('\n{} trained on {} samples.'.format(learner.__class__.__name__, sample_size))  
    
    return results
    

# TODO:        Download data
# =======================================================

data = pd.read_csv('D:/dev/VK_hack/data_charity/finding_donnors_for_charityML/census.csv')

# Success - Display the first record
display('\nDisplay test: \n', data.head(n=1))

n_records = data.shape[0]  
n_greater_50k = np.sum(data['income'] == '>50K')
n_at_most_50k = np.sum(data['income'] == '<=50K')
greater_percent = round((n_greater_50k/(n_greater_50k+n_at_most_50k))*100,2)
# data['income'].value_counts()

print('Total number of records: {}'.format(n_records))
print('Individuals making more than $50,000: {}'.format(n_greater_50k))
print('Individuals making at most $50,000: {}'.format(n_at_most_50k))
print('Percentage of individuals making more than $50,000: {}%'.format(greater_percent))


# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
# vs.distribution(data)



# TODO:      Normilizing and one-hot encodig
# =======================================================

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
# vs.distribution(features_log_transformed, transformed = True)



# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 10))


# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# Encode the 'income_raw' data to numerical values
income = income_raw.map({'<=50K' :0 , '>50K': 1})

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print('{} total features after one-hot encoding.'.format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print('\nEncoded: \n', encoded)



# TODO:       Split data to train and test datasets
# =======================================================


# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

print('\nTraining set has {} samples.'.format(X_train.shape[0]))
print('Testing set has {} samples.'.format(X_test.shape[0]))



# TODO:            Metrics
# =======================================================

#  Ручной расчет F1-меры

'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
#  Calculate accuracy, precision and recall
TP = np.sum(income)
FP = income.count() - TP
TN = 0
FN = 0

accuracy = TP / (TP + FP)
recall = TP / (TP + FN)
precision = TP / (TP + FP)

# Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = ((1 + beta**2) * (precision * recall)) / (((beta**2) * precision) + recall)

print('\nNaive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]\n'.format(accuracy, fscore))



# TODO:            Models
# =======================================================

clf_A = GradientBoostingClassifier(random_state=1)
clf_B = SVC(random_state=1)
clf_C = LogisticRegression(random_state=1)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
# samples_100 is the entire training set i.e. len(y_train)
# samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)

samples_100 = len(y_train)
samples_10 = int(len(y_train) * 0.1)
samples_1 = int(len(y_train) * 0.01)

# Collect results on the learners
results = {}

for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# sns.histplot(results, accuracy, fscore)


# TODO:            Visualisations
# =======================================================

age_plot = data['age'].value_counts().head(7)
age_plot.plot.bar(figsize=(8,6), zorder=100, alpha=0.7, color='g')
plt.grid(ls=':')
plt.show()


# TODO:            Results
# =======================================================

df_results = pd.DataFrame.from_dict(results)
df_results.to_csv(r'D:/dev/VK_hack/data_charity/results.csv')



# TODO:            Indexies of charity humans
# =======================================================

char_hum = y_test.loc[y_test == 1]
ind_char_hum = list(char_hum.index)

print('\nПример первого чела из спрогноженных, его данные: \n', data.iloc[39744])


chel_list = []
for i in range(len(ind_char_hum)):
    chel_list.append(list(data.iloc[ind_char_hum[i]]))


col_list = list(data.columns)
cheliki = pd.DataFrame(data=chel_list, columns=col_list)

cheliki.to_csv(r'D:/dev/VK_hack/data_charity/charity_cheliki.csv')
