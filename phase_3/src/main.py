import glob
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def search(process):
    if process.upper() == 'EDA':
        dir = os.listdir('.\\static\\Visualized_Results\\EDA\\')
        if len(dir) == 0:
            return 0
        else:
            return 1
    else:
        Image = os.listdir('.\\static\\Visualized_Results\\Training\\'+process.upper())
        # Store configuration file values
        if len(Image) >0:
            print("cache found")
            return 1
        else:
            return 0

def searchTraining(stockName):
    try:
        Image = open('static/stocks/'+stockName+'/'+stockName+'1.png', 'r')
        # Store configuration file values
        print("cache found")
        return 1

    except FileNotFoundError:
        # Keep preset values
        print("file not found")
        return 0

#Function to perform and save EDA results
def EDA(df):
    #EDA--------------
    if len(os.listdir('.\\static\\Visualized_Results\\EDA\\')) == 0:
    # print the first 5 rows of the dataset
        print(df.head())
        # get information about the dataset
        print(df.info())
        # check for missing values
        print(df.isnull().sum())
        # get descriptive statistics for the numerical variables
        print(df.describe())

        # visualize the distribution of numerical variables
        # fig1, ax1 = plt.subplots()
        plt.figure(1,figsize=(12,8))
        sns.histplot(df['lead_time'])
        plt.title('Distribution of Lead Time')
        plt.xlabel('Lead Time')
        # plt.show()
        plt.savefig('static/Visualized_Results/EDA/distribution of numerical variables.png')
       
        
        try:
            # visualize the correlation between numerical variables
            plt.figure(2,figsize=(12,8))
            df['hotel'] = df['hotel'].map({"Resort Hotel" :1, "City Hotel":0})
            encoder=LabelEncoder()
            dict_df={}
            for feature in df.columns:
                dict_df[feature]=encoder.fit_transform(df[feature])
            #converting back the encoded feature into dataframe
            df=pd.DataFrame(dict_df)
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.savefig('static/Visualized_Results/EDA/correlation heatmap.png')
        except:
            pass
        

        # explore the distribution of categorical variables
        plt.figure(3, figsize=(12,8))
        sns.countplot(x='hotel', hue='is_canceled', data=df)
        plt.title('Cancellation by Hotel Type')
        plt.xlabel('Hotel Type')
        plt.ylabel('Number of Bookings')
        plt.savefig('static/Visualized_Results/EDA/distribution of categorical variables.png')
        

        # explore the relationship between categorical and numerical variables
        plt.figure(4, figsize=(12,8))
        sns.boxplot(x='hotel', y='lead_time', hue='is_canceled', data=df)
        plt.title('Cancellation by Hotel Type and Lead Time')
        plt.xlabel('Hotel Type')
        plt.ylabel('Lead Time')
        plt.savefig('static/Visualized_Results/EDA/categorical numerical variables relationship.png')
        

        # explore the average daily rate of the bookings
        plt.figure(5, figsize=(12,8))
        sns.lineplot(x='arrival_date_month', y='adr', hue='hotel', data=df)
        plt.title('Average Daily Rate by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Daily Rate')
        plt.savefig('static/Visualized_Results/EDA/daily rate of the bookings.png')
        

        # explore the number of bookings over time
        plt.figure(6, figsize=(12,8))
        sns.lineplot(x='arrival_date_month', y='arrival_date_year', hue='hotel', data=df)
        plt.title('Number of Bookings Over Time')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.savefig('static/Visualized_Results/EDA/bookings over time.png')
        
    toReturn = ['EDA', 'bookings over time','daily rate of the bookings','categorical numerical variables relationship','correlation heatmap','distribution of numerical variables']
    return toReturn

# Function to perform data cleaning and passing ahead for further training
def KNN_train(df):
    
    # Model training -------------------------------
    
    # KNN Cancellation Prediction
    df['hotel'] = df['hotel'].map({"Resort Hotel" :1, "City Hotel":0})
    print(df['arrival_date_month'].unique())
    encoder=LabelEncoder()
    dict_df={}
    for feature in df.columns:
        dict_df[feature]=encoder.fit_transform(df[feature])
    #converting back the encoded feature into dataframe
    df=pd.DataFrame(dict_df)
    print(df['arrival_date_month'].unique())


    # Scaling using MinMaxScaler
    scaler=MinMaxScaler()
    df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

    df_new = df[['hotel','is_canceled','lead_time','arrival_date_month','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','num_children','babies','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes','deposit_type','days_in_waiting_list','adr','required_car_parking_spaces']]
    # splitting data into train and test
    y=df_new['is_canceled']
    x = df_new.drop('is_canceled', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

    #running data on knn model
    KNN=KNeighborsClassifier()
    KNN.fit(X_train,y_train)
    y_pred=KNN.predict(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix as heatmap
    sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('static/Visualized_Results/Training/KNN/Cancellation Prediction KNN.png')
    plt.clf()
    plt.close()

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(report)

    print('accuracy_score:{}'.format(accuracy_score(y_test,y_pred)))
    toReturn = [accuracy_score(y_test,y_pred), 'Cancellation Prediction KNN']
    return toReturn


def logReg(df_cleaned):
     # days in waiting list Logistic Regression---------------------- 
    df_logistic = df_cleaned[['hotel','lead_time','arrival_date_month','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes','deposit_type','days_in_waiting_list','adr','required_car_parking_spaces']]
    df_logistic.loc[df_logistic['days_in_waiting_list'] != 0, 'days_in_waiting_list'] = 1
    
    # splitting data into test and train
    y=df_logistic['days_in_waiting_list']
    x = df_logistic.drop('days_in_waiting_list', axis = 1)
    x_log_train, X_test, y_log_train, y_test = train_test_split(x, y, test_size = 0.30)

    # running it on the model
    lr_model = LogisticRegression()

    # Train the logistic regression model
    lr_model.fit(x_log_train, y_log_train)
    y_pred=lr_model.predict(X_test)
    # Get the coefficients and feature names
    coef = lr_model.coef_[0]
    features = x.columns 

    # accuracy score
    print('accuracy_score:{}'.format(accuracy_score(y_test,y_pred)))

    # Visualize the coefficients
    plt.figure(figsize=(10,6))
    sns.barplot(x=coef, y=features, palette='Blues_d')
    plt.title('Coefficients of Logistic Regression Model')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.savefig('static/Visualized_Results/Training/LR/Days in Waitlist LogReg.png')
    plt.clf()
    plt.close()

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(report)
    toReturn = [accuracy_score(y_test,y_pred), 'Days in Waitlist LogReg']
    return toReturn
    

def NB(df_cleaned):
       
    # Cancellation after waitlist prediction: naive bayes-----------------------------------
    df_nb = df_cleaned[['hotel','is_canceled','lead_time','arrival_date_month','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','babies','num_children','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes','deposit_type','days_in_waiting_list','adr','required_car_parking_spaces']]    
    df_nb = df_nb[df_cleaned['days_in_waiting_list'] != 0]

    # splitting the data into test and train
    y=df_nb['is_canceled']
    x = df_nb.drop('is_canceled', axis = 1)
    x_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

    # xtrain and ytrain are the training data and labels, respectively
    nb_classifier = MultinomialNB()
    nb_classifier.fit(x_train, y_train)

    # assuming xtest is your test data
    y_pred = nb_classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix using a heatmap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('static/Visualized_Results/Training/NB/Cancellation After Waitlist NB.png')
    plt.clf()
    plt.close()
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(report)
    toReturn = [accuracy_score(y_test,y_pred), 'Cancellation After Waitlist NB']
    return toReturn

def DT(df_cleaned):
    # Type of customers DecisionTree-----------------------------------------
    df_dt = df_cleaned[['hotel','lead_time','arrival_date_year','arrival_date_month','arrival_date_day_of_month','stays_in_weekend_nights','customer_type','stays_in_week_nights','adults','babies','num_children','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','booking_changes','deposit_type','adr','required_car_parking_spaces']]
    # add new "family" column based on conditions
    df_dt['family'] = ((df_dt['babies'] + df_dt['num_children'] > 0) & (df_dt['adults'] > 0)).astype(int)
    # splitting data into test and train
    y=df_dt['family']
    x = df_dt.drop('family', axis = 1)
    x_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
    # training the model
    tree=DecisionTreeClassifier()
    tree.fit(x_train,y_train)
    y_pred = tree.predict(X_test)
    print('accuracy_score:{}'.format(accuracy_score(y_test,y_pred)))
    print('f1_score:{}'.format(f1_score(y_test,y_pred)))
    print('precision:{}'.format(precision_score(y_test,y_pred)))
    print('recall:{}'.format(recall_score(y_test,y_pred)))

    # Visualizing the results
    plt.figure(figsize=(20,20))
    plot_tree(tree, feature_names=x.columns, filled=True)
    plt.savefig('static/Visualized_Results/Training/DT/Types of Customers DecisionTree.png')
    plt.clf()
    plt.close()
    toReturn = [accuracy_score(y_test,y_pred), 'Types of Customers DecisionTree']
    return toReturn

def ADA(df_cleaned):
    # Monthly popularity: Adaboost--------------------------------------------------------------
    df_dt = df_cleaned[['hotel','arrival_date_year','arrival_date_month','arrival_date_day_of_month']]
    #splitting data into test and train
    y=df_dt['hotel']
    x = df_dt.drop('hotel', axis = 1)
    x_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30) 
    # Create a Decision Tree Classifier as the base estimator
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Create an AdaBoost classifier with 50 estimators
    ada = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)

    # Fit the AdaBoost classifier to the training data
    ada.fit(x_train, y_train)

    # Predict labels for the testing data
    y_pred = ada.predict(X_test)

    # Calculate evaluation metrics for the AdaBoost classifier
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    # Create a bar plot of feature importances
    feature_importances = ada.feature_importances_
    features = x_train.columns
    plt.bar(features, feature_importances)
    plt.title("Feature Importances")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.savefig('static/Visualized_Results/Training/ADA/Monthly Popularity Adaboost.png')
    plt.clf()
    plt.close()
    toReturn = [accuracy, 'Monthly Popularity Adaboost']
    return toReturn



# Main Stock Prediction Function
def dataCleaningCallingTrain(process):
    #reading the data
    df_hotelbooking = pd.read_csv('hotel_bookings.csv') 
    df = df_hotelbooking

    # data cleaning----------------------
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    # Replace missing values with a default value (e.g. 0)
    df.fillna(0, inplace=True)
    # Remove irrelevant columns
    df.drop(['company'], axis=1, inplace=True)
    # Standardize column names
    df.rename(columns={'children': 'num_children'}, inplace=True)
    df_cleaned = df
    # Detect outliers in a column
    Q1 = df['adr'].quantile(0.25)
    Q3 = df['adr'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df['adr'] < Q1 - 1.5 * IQR) | (df['adr'] > Q3 + 1.5 * IQR)

    # Replace outliers with a default value (e.g. median)
    df.loc[outliers, 'adr'] = df['adr'].median()
    # Remove leading/trailing spaces from string columns
    df['country'] = df['country'].str.strip()
    # Replace inconsistent values in a column
    df['market_segment'] = df['market_segment'].replace('Undefined', 'Other')
    # Remove special characters from string columns
    df['reserved_room_type'] = df['reserved_room_type'].str.replace('/', '')

    # making directory
    path = os.getcwd()
    print(path)

    os.makedirs(path+"\\static\\Visualized_Results\\Training\\KNN", exist_ok=True)
    os.makedirs(path+"\\static\\Visualized_Results\\Training\\ADA", exist_ok=True)
    os.makedirs(path+"\\static\\Visualized_Results\\Training\\DT", exist_ok=True)
    os.makedirs(path+"\\static\\Visualized_Results\\Training\\LR", exist_ok=True)
    os.makedirs(path+"\\static\\Visualized_Results\\Training\\NB", exist_ok=True)
    os.makedirs(path+"\\static\\Visualized_Results\\EDA", exist_ok=True)

    # cleaning for further processes
    df_cleaned['country'] = df_cleaned['country'].str.strip()
    df_cleaned['market_segment'] = df_cleaned['market_segment'].replace('Undefined', 'Other')
    df_cleaned['reserved_room_type'] = df_cleaned['reserved_room_type'].str.replace('/', '')
    encoder=LabelEncoder()
    dict_df_clean={}
    for feature in df_cleaned.columns:
        dict_df_clean[feature]=encoder.fit_transform(df_cleaned[feature])
    #converting back the encoded feature into dataframe
    df_cleaned=pd.DataFrame(dict_df_clean)


    # conditionals to fetch the results
    if process == 'EDA':
        tostore = EDA(df)
    
    # # removing all the results from the training folder
    # files = glob.glob('./static/Visualized_Results/Training/*')
    # for f in files:
    #     os.remove(f)
    
    if process == 'KNN':
        files = glob.glob('./static/Visualized_Results/Training/KNN/*')
        for f in files:
            os.remove(f)
        tostore = KNN_train(df)
    elif process == 'LR':
        files = glob.glob('./static/Visualized_Results/Training/LR/*')
        for f in files:
            os.remove(f)
        tostore = logReg(df_cleaned)
    elif process == 'NB':
        files = glob.glob('./static/Visualized_Results/Training/NB/*')
        for f in files:
            os.remove(f)
        tostore = NB(df_cleaned) 
    elif process == 'DT':
        files = glob.glob('./static/Visualized_Results/Training/DT/*')
        for f in files:
            os.remove(f)
        tostore = DT(df_cleaned)
    elif process == 'ADA':
        files = glob.glob('./static/Visualized_Results/Training/ADA/*')
        for f in files:
            os.remove(f)
        tostore = ADA(df_cleaned)

    
    return tostore
