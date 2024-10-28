'''
This module provides a library of functions to find customers who are likely to churn.
'''

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import pandas as pd
import matplotlib.pyplot as plt


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

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

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
    pass


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
    pass

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
    pass

if __name__ == "__main__":
    print("Importing data")
    dataset = import_data("./data/bank_data.csv")
    print(dataset.head())
    print(dataset.describe())
    
    print("Conducting data exploration")
    perform_eda(dataset)
    
    print("Feature engineering")
    #X_train, X_test, y_train, y_test = perform_feature_engineering(dataset, response='Churn')
    
    print("Model training")
    #train_models(X_train, X_test, y_train, y_test)
    
    print("Training completed")