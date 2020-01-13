import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


_data_connection_prefix = 'sqlite:///'


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath to the database with cleaned data
    
    OUTPUT:
    X - pandas DataFrame with model input data
    Y - pandas DataFrame with model response data
    column names - column names for response data
    '''
    
    engine = create_engine(_data_connection_prefix + database_filepath)
    df = pd.read_sql_table('messages_with_categories', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns


stop_words = stopwords.words("english")


def tokenize(text):
    '''
    INPUT:
    text - message

    OUTPUT:
    tokens - list with all text tokens in the original message
    '''
    
    try:
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        
        clean_tokens = []
        for tok in tokens:
            # lemmatize, normalize case, and remove leading/trailing white space
            clean_tok = lemmatizer.lemmatize(tok.lower().strip())
            if clean_tok not in stop_words:
                clean_tokens.append(clean_tok)

        return clean_tokens
    except Exception as e:
        print(e)
    

def build_model():
    '''
    OUTPUT:
    model - a model which has been optimized for the global task by using GridSearchCV
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    #cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    cv = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=20, cv=5, iid=False, verbose=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - model to be evaluated
    X_test - test input data
    Y_test - test response data
    category_names - column names for response data
    '''
    
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    
    for col in category_names:
        print('Test results for "' + col + '" :\n')
        print(classification_report(Y_test.loc[:, col], Y_pred_df.loc[:, col]))
        print('\n')


def save_model(model, model_filepath):
    '''
    INPUT:
    model - model to be saved as pkl
    model_filepath - filepath and model file name
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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