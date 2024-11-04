'''
This module provides a library of functions to find customers who are likely to churn.
'''

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    
    try:
        dataframe = pd.read_csv(pth, index_col=0)
    except FileNotFoundError:
        print("File not found. Check the path again.")
    except pd.errors.ParserError:
        print("Wrong file format.")
    else:
        # Encode target variable: 0 = Did not churned ; 1 = Churned
        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(lambda value: 0 if value == "Existing Customer" else 1)

        # Drop irrelevant columns
        dataframe.drop('CLIENTNUM', axis=1, inplace=True)
        dataframe.drop('Attrition_Flag', axis=1, inplace=True)
        
        return dataframe


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    
    # Analyze the target variable and plot distribution
    plt.figure(figsize=(10, 6))
    dataframe['Churn'].value_counts().plot(kind='bar')
    plt.title(f"Distribution of Churn")
    plt.xlabel("Churn")
    plt.ylabel("Count")

    # Save plot with a unique filename
    plt.savefig(os.path.join("./images/eda", "Churn_distribution.png"))
    plt.close()
    
    # Analyze categorical features and plot distribution
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns
    for feature in categorical_cols:
        plt.figure(figsize=(10, 6))
        dataframe[feature].value_counts().plot(kind='bar')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.savefig(os.path.join("./images/eda", f"{feature}_distribution.png"))
        plt.close()

    # Analyze Numeric features
    dataframe.describe().to_csv(os.path.join("./images/eda", "numerical_feature_description.csv"))
    
    numerical_cols = list(set(dataframe.columns)-{'Churn'}-set(categorical_cols))
    
    for feature in numerical_cols:
        plt.figure()
        dataframe[feature].plot(kind='hist', bins=30, edgecolor='black')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join("./images/eda", f"{feature}_distribution.png"))
        plt.close()

def encoder_helper(dataframe, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    
    for category in category_lst:
        categories = dataframe.groupby(category).mean()[response]
        dataframe[category + "_response"] = dataframe[category].apply(lambda item: categories.loc[item])
        
    dataframe.drop(category_lst, axis=1, inplace=True)
    
    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    
    # Encoding categorical columns
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns
    dataframe = encoder_helper(dataframe, categorical_cols, response=response)
    
    y = dataframe[response]
    X = dataframe.drop(response, axis=1)
    
    return train_test_split(X, y, test_size=0.3, random_state=0)
    
    

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
             y_train: training response values
             y_test:  test response values
             y_train_preds_lr: training predictions from logistic regression
             y_train_preds_rf: training predictions from random forest
             y_test_preds_lr: test predictions from logistic regression
             y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    
    # Plot Logistic Regression Classification Report
    plt.rc('figure', figsize=(6, 6))
    
    # Classification report for training dataset
    plt.text(0.01, 1.0, str('Logistic Regression Training'))
    plt.text(0.01, 0.1, str(classification_report(y_train, y_train_preds)))

    # Classification report for test dataset
    plt.text(0.01, 0.6, str('Logistic Regression Test'))
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)))
    plt.axis('off')

    # Save figure
    plt.savefig(os.path.join("./images/results", "classification_report_lr.png", bbox_inches='tight'))
    plt.close()
                
    # Plot Random Forest Classification Report
    plt.rc('figure', figsize=(6, 6))
    
    # Classification report for training dataset
    plt.text(0.01, 1.0, str('Logistic Regression Training'))
    plt.text(0.01, 0.1, str(classification_report(y_train, y_train_preds)))

    # Classification report for test dataset
    plt.text(0.01, 0.6, str('Logistic Regression Test'))
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)))
    plt.axis('off')

    # Save figure
    plt.savefig(os.path.join("./images/results", "classification_report_rf.png", bbox_inches='tight'))
    plt.close()

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
                
    # Feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Create figure
    plt.figure(figsize=(25, 5))
    plt.title(f"Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure
    plt.savefig(os.path.join(output_pth, "feature_importance.png"))
    plt.close()

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    
    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=0)

    # grid search for random forest parameters and instantiation
    param_grid = {
        'n_estimators': [100, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rf_classifier = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)

    # Train Ramdom Forest using GridSearch
    cv_rf_classifier.fit(X_train, y_train)

    # Get predictions
    y_train_preds_rf = cv_rf_classifier.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rf_classifier.best_estimator_.predict(X_test)

    # Display feature importance on train data
    feature_importance_plot(cv_rf_classifier.best_estimator_, X_train, "./images/results")
                
    # Save model
    joblib.dump(cv_rf_classifier.best_estimator_, './models/rf_model.pkl')

                
    # Initialize Logistic Regression model
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=3000)
                
    # Train Logistic Regression
    lr_classifier.fit(X_train, y_train)

    y_train_preds_lr = lr_classifier.predict(X_train)
    y_test_preds_lr = lr_classifier.predict(X_test)

    # Save model
    joblib.dump(lr_classifier, './models/lr_model.pkl')

    # Calculate classification scores
    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)


if __name__ == "__main__":
    print("Importing data")
    dataset = import_data("./data/bank_data.csv")
    print(dataset.head())
    print(dataset.describe())
    
    print("Conducting data exploration")
    perform_eda(dataset)
    
    print("Feature engineering")
    X_train, X_test, y_train, y_test = perform_feature_engineering(dataset, response='Churn')
    
    print("Model training")
    train_models(X_train, X_test, y_train, y_test)
    
    print("Training completed")