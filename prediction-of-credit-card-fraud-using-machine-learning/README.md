# prediction-of-credit-card-fraud-using-machine-learning
 
the credit card holds the major part of money transactions. Due to the online shopping growth, transactions using credit cards have been more frequent. Which helps in speed of payments, and it also attracts criminal attention. As improper use of credit/debit cards to make fraudulent transactions, obtaining money and goods. Where the numbers are taken by websites without protection, malicious credit machines, and numbers shared using social media apps.
the machine learning models for fraud detection is a powerful tool.it uses a large quantity of data collected in the past, to verify and classify the new transactions in real-time.
implementing a machine learning model using python, pandas, and scikit-learn to classify new transactions as fraudulent or not.
# About the Dataset
A credit card is one of the most used financial products to make online purchases and 
payments. Though the Credit cards can be a convenient way to manage your finances, they can 
also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit 
card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card 
transactions so that customers are not charged for items that they did not purchase. 
The dataset contains transactions made by credit cards in September 2013 by European 
cardholders. This dataset presents transactions that occurred in two days, where we have 492 
frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) 
account for 0.172% of all transactions.
We have to build a classification model to predict whether a transaction is fraudulent or not.
•	Time - Contains the seconds elapsed between each transaction;
•	Amount - Total transactioned value;
•	Class - Label given from transactions, where 0 represent a normal transaction and 1 reffers to a fraudulent transaction.
# Libraries used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

# Exploratory Analysis
In this section, we’ll make a preliminary data analysis, to verify the variables in the dataset, null values, outliers, and histograms of legal and illegal transactions.

 
# Legal and Illegal Transactions,
There’s a considerable discrepancy between the legal [0] transactions and illegal [1] occurrences. Due to this, the dataset will be balanced before the setup of the machine learning model.
# Checking Outliers in ‘Amount’ values: 
there’s a lot of outliers in the Amount column, we will perform a cleaning of those valueswhich are higherthan upperlimit and lower than lower limit and redo the plot, the values in the Amount column will be less discrepant now.

# Pre-Processing Data
Before we create the machine learning model, we will need to make some adjustments to the dataset values. The steps are listed below:
•	Standardize the Time and Amount values;
•	Divide the dataset into train and test;
•	Balance the values.

# Dividing the dataset in train and test
The train/test split it’s an important step in machine learning process. Where the model is split into two sets: training and testing set. Usually, the major part of the dataset is reserved for the training and a small part for testing. The train data will be used to create the machine learning model, and the test data is used to check the accuracy of the model.

# Balancing dataset
As we can see in the exploratory analysis, we have a discrepancy between the number of legal transactions and frauds. To feed the machine learning model without any bias. We will perform the balance of the dataset to keep the values as close as possible.
•	Random Oversampling: Randomly duplicate examples in the minority class.
•	Random Undersampling: Randomly delete examples in the majority class.

Now, with the balanced dataset, we can proceed with the setup of the machine learning model.

# Machine Learning Model
After pre-processing the data, we can create the machine learning model, as we are dealing with a binary problem, we will use the logistic regression to check if the transactions can be labeled as legal or scam.

# Model Performance
In the report below, we’ve some metrics to check the model performance, let’s make a brief explanation about how to understand those values, and evaluate the machine learning model. Before explaining the mathematical formulas, we will explain some terms and what it does represent.
•	TN — True Negative: when a case was negative and predicted negative;
•	TP — True Positive: when a case was positive and predicted positive;
•	FN — False Negative: when a case was positive but predicted negative;
•	FP — False Positive: when a case was negative but predicted positive.
Accuracy — What percent of predictions the model did correctly?
 
Precision — What percent of the predictions were correct?
 
Recall — What percent of the positive cases the model catch?
 
F1-Score — What percent of positive predictions were correct?
 
# classification report
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, text_fontsize='small', title_fontsize='medium', cmap='Blues');
print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))Classification Report:
               precision    recall  f1-score   support

           0      1.000     0.990     0.995     63123
           1      0.130     0.920     0.228       100

    accuracy                          0.990     63223
   macro avg      0.565     0.955     0.611     63223
weighted avg      0.998     0.990     0.994     63223
 

# Conclusion
The use of technologies in credit card fraud detection using machine learning models, shows a powerful tool, once the models receive huge quantities of new data every day. Although we have reached good results in the model, it’s important to make tests in new databases, to observe and improve the model performance.
