#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:49:52 2024

@author: ryanpatil
"""

### Import Python Packages ###
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import learning_curve

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, auc, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

############################################################################################################

### Define Parameters and Variables for plots ###
font_size = 20
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size + 2
plt.rcParams['xtick.labelsize'] = font_size - 2
plt.rcParams['ytick.labelsize'] = font_size - 2
plt.rcParams['legend.fontsize'] = font_size - 2

plot_colors = ['#00A5E0', '#DD403A']
plot_colors_cat = ['#E8907E', '#D5CABD', '#7A6F86', '#C34A36', '#B0A8B9', '#845EC2', '#8f9aaa', '#FFB86F', '#63BAAA', '#9D88B3', '#38c4e3']
plot_colors_comp = ['steelblue', 'seagreen', 'black', 'darkorange', 'purple', 'firebrick', 'slategrey']

random_state = 42
scoring_metric = 'recall'
comparison_dict, comparison_test_dict = {}, {}

############################################################################################################

# Define Plotting Functions

def continuous_variable_plot(train_df, df_retained, df_churned, feature):
    
    '''Generates a boxplot and histogram to visualise the churned and retained distributions for the given feature. '''
    
    # Copy the training dataframe and specify the 'Exited' column as a categorical data type
    df_boxplot = train_df.copy()
    df_boxplot['Exited'] = df_boxplot['Exited'].astype('category')
    
    # Create a plot with subplots, one for the histogram and one for the boxplot
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': (.8, .4)})
    
    # Plot histogram with specified settings
    for df, color, label in zip([df_retained, df_churned], plot_colors, ['Retained', 'Churned']):
        sns.histplot(data=df, x=feature, bins=20, color=color, alpha=0.67, edgecolor='black', label=label, kde=False, ax=ax1)
        
    # Show legend
    ax1.legend()
    
    # Generate boxplot for the retained and churned customers with regard to the specified feature
    sns.boxplot(x=feature, y='Exited', data=df_boxplot, palette=plot_colors, ax=ax2)
    ax2.set_ylabel('')
    ax2.set_yticklabels(['Retained', 'Churned'])
    
    # Show plots
    plt.tight_layout()
    plt.show()
    

def categorical_variable_plot(train_df, feature):
    
    '''Generates a countplot to show the frequency of each categorical feature and a bar chart for the churn rate.'''
    
    # Create a plot with subplots for each chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Generate countplot for the given feature with respect to whether a customer was retained or churned
    sns.countplot(x=feature, hue='Exited', data=train_df, palette=plot_colors, ax=ax1)
    
    # Set label and show legend
    ax1.set_ylabel('Count')
    ax1.legend(labels=['Retained', 'Churned'])
    
    # Generate a bar chart for each feature to show the churn rate
    sns.barplot(x=feature, y='Exited', data=train_df, palette=plot_colors_cat, ax=ax2)
    
    # Set labels
    ax2.set_ylabel('Churn rate')
    
    # If the feature is HasCrCard or IsActiveMember, specify either yes or no for clarity
    if (feature == 'HasCrCard' or feature == 'IsActiveMember'):
        ax1.set_xticklabels(['No', 'Yes'])
        ax2.set_xticklabels(['No', 'Yes'])
        
    # Show plots
    plt.tight_layout()
    plt.show()
    
    
def confusion_matrix_plot(conf_matrix, ax):
    
    '''Create a confusion matrix to visualise the performance of a classifier'''
    
    # Generate a heatmap style confusion matrix 
    sns.heatmap(data=conf_matrix, annot=True, cmap='Blues', annot_kws={'fontsize': 32}, ax=ax)
    
    # Set labels 
    ax.set_xlabel('Predicted Label')
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Retained', 'Churned'])

    ax.set_ylabel('True Label')
    ax.set_yticks([0.25, 1.25])
    ax.set_yticklabels(['Retained', 'Churned']);


def learning_curve_plot(estimator, X, y, ax, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    
    '''Plot learning curves for a particular classifier.'''
    
    # Generate outputs to track the progress of classifier learning
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy')
    
    # Obtain the average and standard deviation of the training scores and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Fill space between curves
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.3, color='blue')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.3, color='orange')
    
    # Plot curves to show the training and cross-validation scores
    ax.plot(train_sizes, train_scores_mean, color='blue', marker='o', linestyle='-', label='Training Score')
    ax.plot(train_sizes, test_scores_mean, color='orange', marker='o', linestyle='-', label='Cross-validation Score')
    
    # Set labels and legend for plot
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Score')
    ax.legend(loc='best', fontsize=16);


def classifier_performance(X_train, y_train, classifier, classifier_name, classifier_name_abv):
    
    '''Assess the overall performance of each classifier using recall as the scoring metric and display results.'''
    
    # Print classifier name, highest recall score and optimal parameters
    print('\n', classifier_name)
    print('-------------------------------')
    print('   Best Score ({}): '.format(scoring_metric) + str(np.round(classifier.best_score_, 3)))
    print('   Best Parameters: ')
    for key, value in classifier.best_params_.items():
        print('      {}: {}'.format(key, value))
        
    # Generate cross-validated predictions for the classifier and perform rounding
    y_pred_pp = cross_val_predict(estimator=classifier.best_estimator_, X=X_train, y=y_train, cv=5, method='predict_proba')[:, 1]
    y_pred = y_pred_pp.round()
    
    # Create a confusion matrix to compare the true labels with the predicted labels
    cm = confusion_matrix(y_train, y_pred, normalize='true')
    
    # Generate an ROC curve to assess classifier performance with the true positive rate against the false positive rate 
    fpr, tpr, _ = roc_curve(y_train, y_pred_pp)
    
    # Create a key, value pair in the predefined dictionary to record classifier performance metrics
    comparison_dict[classifier_name_abv] = [accuracy_score(y_train, y_pred), precision_score(y_train, y_pred), recall_score(y_train, y_pred), roc_auc_score(y_train, y_pred_pp), fpr, tpr]
    
    # Create a plot with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot confusion matrix and learning curve for classifier
    confusion_matrix_plot(cm, ax1)
    learning_curve_plot(classifier.best_estimator_, X_train, y_train, ax2)

    plt.tight_layout();


############################################################################################################

### Basic Data statistics ###

# Determine path to data files relative to this script
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Load training set
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), encoding='utf-8')

# Check the number of rows and columns
print('Training set contains {} rows and {} columns.'.format(train_df.shape[0], train_df.shape[1]))

# Check number of duplicate rows
num_duplicates = train_df.duplicated().sum()
print("Number of duplicate rows:", num_duplicates)

# Drop unnecessary columns
train_df.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
print(train_df.columns)

# Obtain initial information regarding training set nulls/non-nulls
train_df.info()

# Print summary statistics of training set and store as variable
summary_stats = train_df.describe().T
print(summary_stats)

### Exploratory Data Analysis ###

# Create a bar chart to show the percentage of customers that churned versus
# were retained by the bank in our training set

fig, ax = plt.subplots(figsize=(6, 6))

sns.countplot(x='Exited', data=train_df, palette=plot_colors, ax=ax)

for index, value in enumerate(train_df['Exited'].value_counts()):
    label = '{}%'.format(round((value / train_df['Exited'].shape[0]) * 100, 2))
    ax.annotate(label,xy=(index, value + 5000), ha='center', va='center', color=plot_colors[index],
                fontweight='bold', size=font_size + 4)

ax.set_xticklabels(['Retained', 'Churned'])
ax.set_xlabel('Status')
ax.set_ylabel('Count')
ax.set_ylim([0, 150000]);

# Specify which columns in our dataset are categorical and which are numerical
continuous = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categorical = ['Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']

print('Continuous: ', ', '.join(continuous))
print('Categorical: ', ', '.join(categorical))

# Plot a histogram for each of the continuous variables
train_df[continuous].hist(figsize=(12, 10), bins=20,layout=(2, 2), color='firebrick',
                          edgecolor='black', linewidth=1.5);

# Check correlations between each pair of numerical variables
fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(train_df[continuous].corr(), annot=True, annot_kws={'fontsize': 12}, cmap='Blues',ax=ax)

ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=45)

plt.tight_layout()

# Separate training set based upon whether the customer churned or not
df_churned = train_df[train_df['Exited'] == 1]
df_retained = train_df[train_df['Exited'] == 0]

### Explore the effect of the numerical variables on customer churn ###

# Examine the effect of age on churn
continuous_variable_plot(train_df, df_retained, df_churned, 'Age')

# Examine the effect of credit score on churn
continuous_variable_plot(train_df, df_retained, df_churned, 'CreditScore')

# Examine the effect of balance on churn
continuous_variable_plot(train_df, df_retained, df_churned, 'Balance')

# Examine the effect of estimated salary on churn
continuous_variable_plot(train_df, df_retained, df_churned, 'EstimatedSalary')

# Explore the counts of the categorical variables in the training set
df_cat = train_df[categorical]

# Generate a countplot for visualisation
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

for index, column in enumerate(df_cat.columns):

    plt.subplot(2, 3, index + 1)
    sns.countplot(x=column, data=train_df, palette=plot_colors_cat)

    plt.ylabel('Count')
    if (column == 'HasCrCard' or column == 'IsActiveMember'):
        plt.xticks([0, 1], ['No', 'Yes'])

plt.tight_layout();

# Examine the effect of geography on churn
categorical_variable_plot(train_df, 'Geography')

# Examine the effect of gender on churn
categorical_variable_plot(train_df, 'Gender')

# Examine the effect of tenure on churn
categorical_variable_plot(train_df, 'Tenure')

# Examine the effect of number of products on churn
categorical_variable_plot(train_df, 'NumOfProducts')

# Examine the effect of having a credit card on churn
categorical_variable_plot(train_df, 'HasCrCard')

# Examine the effect of a customer being an active member on churn
categorical_variable_plot(train_df, 'IsActiveMember')

############################################################################################################

### Data Pre-Processing for Machine Learning Task ###

# Create new features

# 1st Attribute - Balance Salary Ratio
train_df['BalanceSalaryRatio'] = train_df.Balance/train_df.EstimatedSalary

#  2nd Attribute-Tenure  Age
train_df['TenureByAge'] = train_df.Tenure/(train_df.Age)

# 3rd Attribute- Credit Score Given Age
train_df['CreditScoreByAge'] = train_df.CreditScore/(train_df.Age)

# Label encode gender categorical variable
train_df['Gender'] = LabelEncoder().fit_transform(train_df['Gender'])

# As there is minimal difference in churn rate between French and Spanish customers, they can be combined 
# into a single category represented by a 0. I.e. Geography becomes German - 1, Non-German - 0
train_df['Geography'] = train_df['Geography'].map({'Germany': 1,'Spain': 0,'France': 0})

# Instantiate standard scaler
scaler = StandardScaler()

# Scale all continous variables in training set
scl_columns = ['CreditScore', 'Age', 'Balance', 'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreByAge']
train_df[scl_columns] = scaler.fit_transform(train_df[scl_columns])

# Separate training set into training features and training labels
y_train = train_df['Exited']
X_train = train_df.drop('Exited', axis=1)

# Compute value counts for labels
y_train.value_counts()

# Perform class balancing of the training set using the SMOTE approach
over = SMOTE(sampling_strategy='auto', random_state=random_state)
X_train, y_train = over.fit_resample(X_train, y_train)

# Re-compute value counts for labels
y_train.value_counts()

############################################################################################################

### Begin initial machine learning model training ###

# Specify simple models to provide an initial estimate of classifier performance
initial_classifier_list = [('Gaussian Naive Bayes', GaussianNB()), ('Logistic Regression', LogisticRegression(random_state=random_state))]

# For each initial classifier, perform 5-fold cross-validation and estimate performance based on recall
cv_base_mean, cv_std = [], []
for initial_classifier in initial_classifier_list:
    
    # Compute cross-validation score
    cv = cross_val_score(estimator=initial_classifier[1], X=X_train, y=y_train, scoring=scoring_metric, cv=5, n_jobs=-1)
    
    # Append results to output lists
    cv_base_mean.append(cv.mean())
    cv_std.append(cv.std())

print('Initial Models (Recall):')

# Show average recall score for each classifier
for i in range(len(initial_classifier_list)):
    print('   {}: {}'.format(initial_classifier_list[i][0], np.round(cv_base_mean[i], 2)))
    
############################################################################################################

### Begin machine learning model training with hyperparameter tuning ###

##### LOGISTIC REGRESSION #####

# Instatiate Logistic Regression model
lr = LogisticRegression(random_state=random_state)

# Specify parameters over which to opimtise model
param_grid = {'max_iter': [100],'penalty': ['l1', 'l2'],'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],'solver': ['lbfgs', 'liblinear']}

# Perform hyperparameter optimisation using grid search approach
lr_clf = GridSearchCV(estimator=lr, param_grid=param_grid, scoring=scoring_metric, cv=5, verbose=False, n_jobs=-1)

# Fit model with optimal hyperparameters to training set
best_lr_clf = lr_clf.fit(X_train, y_train)

# Assess classifier performance
classifier_performance(X_train, y_train, best_lr_clf, 'Logistic Regression', 'LR')

##### RANDOM FOREST CLASSIFIER #####

# Instantiate Random Forest Model
rf = RandomForestClassifier(random_state=random_state)

# Specify parameters over which to optimise model
param_grid = {'n_estimators': [100],'criterion': ['entropy', 'gini'],'bootstrap': [True, False],'max_depth': [6],
              'max_features': ['auto', 'sqrt'],'min_samples_leaf': [2, 3, 5], 'min_samples_split': [2, 3, 5]}

# Perform hyperparameter optimisation using grid search approach
rf_clf = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scoring_metric, cv=5, verbose=False, n_jobs=-1)

# Fit model with optimal hyperparameters to training set
best_rf_clf = rf_clf.fit(X_train, y_train)

# Assess classifier performance
classifier_performance(X_train, y_train, best_rf_clf, 'Random Forest', 'RF')

##### GRADIENT BOOSTING CLASSIFIER #####

# Instatiate Gradient Boosting model
gbc = GradientBoostingClassifier(random_state=random_state)

# Specify parameters over which to opimtise model
param_grid = {'n_estimators': [600],'subsample': [0.66, 0.75],'learning_rate': [0.001, 0.01],'max_depth': [3],'min_samples_leaf': [3, 5],
              'min_samples_split': [5, 7], 'max_features': ['auto', 'log2', None],'n_iter_no_change': [20],'validation_fraction': [0.2],'tol': [0.01]}

# Perform hyperparameter optimisation using grid search approach
gbc_clf = GridSearchCV(estimator=gbc, param_grid=param_grid, scoring=scoring_metric, cv=5, verbose=False, n_jobs=-1)

# Fit model with optimal hyperparameters to training set
best_gbc_clf = gbc_clf.fit(X_train, y_train)

# Assess classifier performance
classifier_performance(X_train, y_train, best_gbc_clf, 'Gradient Boosting Classifier', 'GBC')
best_gbc_clf.best_estimator_.n_estimators_

##### XGBOOST CLASSIFIER #####

# Instatiate XGBoost Classifier model
xgb = XGBClassifier(random_state=random_state)

# Specify parameters over which to optimise model
param_grid = {'n_estimators': [50],'learning_rate': [0.001, 0.01],'max_depth': [3, 4],'reg_alpha': [1, 2],'reg_lambda': [1, 2],
              'subsample': [0.5, 0.75],'colsample_bytree': [0.50, 0.75],'gamma': [0.1, 0.5, 1],'min_child_weight': [1]}

# Perform hyperparameter optimisation using grid search approach
xgb_clf = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=scoring_metric, cv=5, verbose=False, n_jobs=-1)

# Fit model with optimal hyperparameters to training set
best_xgb_clf = xgb_clf.fit(X_train, y_train)

# Assess classifier performance
classifier_performance(X_train, y_train, best_xgb_clf, 'XGBoost Classifier', 'XGB')

############################################################################################################

### Begin ensemble learning approach with soft voting procedure ###

# Specify estimators/models to be included in ensemble
estimators = [('LR', best_lr_clf.best_estimator_), ('RF', best_rf_clf.best_estimator_),
              ('GBC', best_gbc_clf.best_estimator_), ('XGB', best_xgb_clf.best_estimator_)]

# Instantiate voting classifier with specified parameters
tuned_voting_soft = VotingClassifier(estimators=estimators[1:], voting='soft', n_jobs=-1)

# Append soft voting classifier to list of estimators
estimators.append(('SoftV', tuned_voting_soft))

# Compute cross-validated predictions on training set using 5-fold cross validation
y_pred_pp = cross_val_predict(tuned_voting_soft,X_train,y_train,cv=5,method='predict_proba')[:, 1]

# Round predicted labels
y_pred = y_pred_pp.round()

# Generate confusion matrix for ensemble classifier to visualise performance
cm = confusion_matrix(y_train, y_pred, normalize='true')

# Generate roc curve for ensemble classifier to visualise performance
fpr, tpr, _ = roc_curve(y_train, y_pred_pp)

# Record performance metrics of ensemble classifier in dictionary
comparison_dict['SVot'] = [accuracy_score(y_train, y_pred), precision_score(y_train, y_pred),
                            recall_score(y_train, y_pred), roc_auc_score(y_train, y_pred_pp), fpr, tpr]

# Show recall score of ensemble model
print('Soft Voting\n-----------------')
print('  Recall: ', np.round(recall_score(y_train, y_pred), 3))

# Plot confusion matrix and learning curves for ensemble classifier
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
confusion_matrix_plot(cm, ax1)
learning_curve_plot(tuned_voting_soft, X_train, y_train, ax2)

############################################################################################################

### Predict class labels for each instance in the test set, using machine learning model individually
# and then together in an ensemble ###

# Load test set and store as a dataframe
X_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), encoding='utf-8')

# Perform pre-processing steps on test set
X_test.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
X_test['Gender'] = LabelEncoder().fit_transform(X_test['Gender'])
X_test['Geography'] = X_test['Geography'].map({'Germany': 1,'Spain': 0,'France': 0})

# 1st Attribute - Balance Salary Ratio
X_test['BalanceSalaryRatio'] = X_test.Balance/X_test.EstimatedSalary

# 2nd Attribute-Tenure  Age
X_test['TenureByAge'] = X_test.Tenure/(X_test.Age)

# 3rd Attribute- Credit Score Given Age
X_test['CreditScoreByAge'] = X_test.CreditScore/(X_test.Age)

# Scale test set
X_test[scl_columns] = scaler.transform(X_test[scl_columns])

# Create output dictionary for test predictions
model_perf_test_set = {}

# Obtain individual model test predictions
lr_test_predictions = best_lr_clf.best_estimator_.predict(X_test)
model_perf_test_set['Logistic Regression Classifier'] = lr_test_predictions

rf_test_predictions = best_rf_clf.best_estimator_.predict(X_test)
model_perf_test_set['Random Forest Classifier'] = rf_test_predictions

gbc_test_predictions = best_gbc_clf.best_estimator_.predict(X_test)
model_perf_test_set['Gradient Boosting Classifier'] = gbc_test_predictions

xgb_test_predictions = best_xgb_clf.best_estimator_.predict(X_test)
model_perf_test_set['XGBoost Classifier'] = xgb_test_predictions

# Obtain ensemble model test predictions
tuned_voting_soft.fit(X_train, y_train)

######## These are the final predictions for the ensemble model if you are using this as the final output ########
ensemble_predictions = tuned_voting_soft.predict(X_test)
#################################################################################################################

model_perf_test_set['Soft Voting Ensemble Classifier'] = ensemble_predictions 

# Plot ROC Curve for Ensemble Classifier
fig, ax = plt.subplots(figsize=(10, 5))

for index, key in enumerate(comparison_dict.keys()):
    auc, fpr, tpr = comparison_dict[key][3], comparison_dict[key][4], comparison_dict[key][5]
    ax.plot(fpr, tpr, color=plot_colors_comp[index], label='{}: {}'.format(key, np.round(auc, 3)))

ax.plot([0, 1], [0, 1], 'k--', label='Baseline')

ax.set_title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_ylabel('True Positive Rate')
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.autoscale(axis='both', tight=True)
ax.legend(fontsize=14);
