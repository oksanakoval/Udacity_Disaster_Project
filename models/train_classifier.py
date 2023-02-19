import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet'])
nltk.download('omw-1.4')

regex_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):    
    '''
    INPUT 
        database_filepath - filpath to import messages databes     
    OUTPUT
        Returns the following variables:
        X - Returns the input features
        Y - Returns the categories in the dataset 
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("msg_table", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    return X, Y, df
    
    
def tokenize(text):   
    '''
    INPUT 
        text: Text to be processed   
    OUTPUT
        Returns a processed text variable that was tokenized, lower cased, stripped, and lemmatized. 
        URLs replaced with a placeholder 
    '''
    detect_urls = re.findall(regex_url, text)
    for url in detect_urls:
        text = text.replace(url, 'linkURL')
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens
    

def build_model():  
    '''
    OUTPUT
        Returns a pipeline model that has gone through tokenization, count vectorization, 
        TFIDTransofmration and created into a ML model
    '''
    # building pipeline to build a model

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
      
    # parameters to improve model
    parameters = {
        'tfidf__norm':['l1'],
        'clf__estimator__min_samples_split':[2],
    }

    model = GridSearchCV(pipeline, parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT 
        model: model to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories 
    OUTPUT
        Prints out the precision, recall and f1-score
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=Y_test.keys()))
    

def save_model(model, model_filepath):
    '''
    INPUT 
        model: the model to be saved
        model_filepath: filepath to the model to be saved
    OUTPUT
        saves the model to disk
    '''    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()