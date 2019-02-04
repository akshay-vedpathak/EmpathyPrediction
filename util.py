import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import Lasso
import operator
import numpy

'''
Function that loads the data from the csv file
'''
def load_data():
    print("Loading data from responses.csv")
    data = pd.read_csv("responses.csv")
    return data

'''
Function that handles missing values in data and returns the cleaned feature set for both numeric and categorical values
It also applies LabelEncoding on categorical features
'''
def preprocess_data(data):
    categorical = []
    numeric = []
    for column in data:
        if data[column].dtype == 'object':
            categorical.append(column)
        else:
            numeric.append(column)
    print("Identified "+str(len(numeric))+" numeric attributes from the dataset")
    print(numeric)
    print("Identified "+str(len(categorical))+" categorical attributes from the dataset")
    print(categorical)
    numeric_data = data[numeric]
    df_categorical = data[categorical]
    
    numeric_features = numeric_data.fillna(numeric_data.mean())
    numeric_features = numeric_features.astype(int)
    
    df_categorical = df_categorical.astype(str)
    modes = {}
    for column in df_categorical:
        line = str(df_categorical[column].mode())
        line = line.replace('0','')
        line = line.replace('dtype','')
        line = line.replace(':','')
        line = line.replace('object','')
        line = line.lstrip().rstrip()
        modes.update({column:line})
    
    categorical_data = []
    for column in df_categorical:
        df_categorical[column] = df_categorical[column].astype('category')
        categorical_data.append(df_categorical[column].fillna(value = modes.get(column)))

    categorical_features = pd.DataFrame(categorical_data) 
    categorical_features = categorical_features.transpose()
    categorical_features.columns = categorical
    le = LabelEncoder()
    label_encodings = categorical_features.apply(le.fit_transform)
    label_encodings = label_encodings.astype(int)
    
    preprocessed_data = pd.concat([numeric_features,label_encodings],axis=1)
    print("Returning preprocessed dataset")
    return preprocessed_data
    
'''
The function takes the entire data and label to extract from it, and returns the data splitted into features and labels
'''
def splitFeaturesAndLabel(data,label):
    print("Setting Label to "+label+" for classification")
    labels = data[label]
    print("Dropping the label column from feature set")
    data = data.drop(label,axis=1)
    print("Returning "+str(data.shape[1])+" features and label = "+label+" for classification")
    return data,labels

'''
The function takes X_train, Y_train and X_test
Then it uses three classifiers(ExtraTreesClassifier, RandomForestClassifier and Lasso) to get ranks for each feature from the X_train and Y_train, using Recursive Feature Elimination. It thens sorts the indexes of the features based on the sum of the ranks obtained for each feature using the different classifiers,it from the sorted feature list it selects a subset of the features based on a threshold value to ensure that we do end up getting the best features. It then also transforms the X_train and X_test to only use the best features we have identified
'''
def getBestFeatures(X_train,Y_train,X_test):
    model = RandomForestClassifier()
    rfe = RFE(model)
    rfe.fit(X_train,Y_train)
    print("Getting RFE rankings for RandomForestClassifier")
    rfe1_ranking = rfe.ranking_

    model = ExtraTreesClassifier()
    rfe = RFE(model)
    rfe.fit(X_train,Y_train)
    print("Getting RFE rankings for ExtraTreesClassifier")
    rfe2_ranking = rfe.ranking_

    model = Lasso()
    rfe = RFE(model)
    rfe.fit(X_train,Y_train)
    print("Getting RFE rankings for Lasso")
    rfe3_ranking = rfe.ranking_
    
    feature_ranks = {}
    for i in range(len(rfe1_ranking)):
        feature_ranks.update({i:rfe1_ranking[i]+rfe2_ranking[i]+rfe3_ranking[i]})
        
    sorted_ranks = sorted(feature_ranks.items(), key=operator.itemgetter(1))
    important_features = []
    for element in sorted_ranks:
        if element[1]<=70:
            important_features.append(element[0])
 
    column_list = [column for column in X_train.columns]
    best_features = [column_list[i] for i in important_features]
    print("Extracted "+str(len(best_features))+" best features from the training data")
    print("Transforming X_train and X_test to only include the best features picked")
    X_train = X_train[best_features]
    X_test = X_test[best_features]
    return X_train,X_test
    
'''
The function takes the model, X, Y and params to perform GridSearchCV on the model to identify the best parameters over 5-fold crossvalidation and returns the best hyperparameters identified
'''
def getBestHyperParameters(model,X,Y,params):
    print("Performing grid search for model: "+str(model))
    print("With hyper parameter options: "+str(params))
    grid_search = GridSearchCV(model,params,cv=5,scoring='accuracy')
    grid_search.fit(X,Y)
    print("Returning best params: "+str(grid_search.best_params_))
    return grid_search.best_params_

'''
Function that takes the model along with best hyperparameters, trains the model on the data and gives predictions
'''   
def trainAndPredict(X_train,Y_train,model,X_test):
    print("Training on model: "+str(model))
    model.fit(X_train,Y_train)
    print("Predicting for test data:")
    preds = model.predict(X_test)
    print("Returning predictions")
    return preds

'''
Function that evaluates the predictions and prints out the accuracy and classification report
'''
def printAccuracyAndClassficationReport(preds,Y_test,classes):
    print("Accuracy for predictions: "+str(accuracy_score(Y_test,preds)))
    print("Classification Report:")
    print(classification_report(Y_test,preds,target_names = classes))