import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath, table_name):
    """
    This function loads data from sqlite database and defines category 
    names, feature variables and target variables.
    params:
    database_filepath: sqlite database filename
    table_name: sqlite database table name
    Returns feature variables, target variables and category names
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, engine)
    category_names = df.columns[4:40]

    # feature variable
    X = df.message.values

    # target variables
    y = df[category_names].values

    return X, y, category_names


def tokenize(text):
    """
    This function tokenizes the text, lemmatizes, normalizes case, and 
    removes leading/trailing white space.
    params:
    text: text to be processed
    Returns cleaned tokens list
    """
    # tokenize the text
    tokens = word_tokenize(text)

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize, normalize case, and remove leading/trailing white space
    clean_tokens = [lemmatizer().lemmatize(x.lower().strip()) for x in tokens]
    
    return clean_tokens


def build_model():
    """
    This function builds machine learning pipeline that processes text and then performs 
    multi-output classification on the 36 categories in the dataset. 
    GridSearchCV is used to find the best parameters for the model.
    Returns the model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'vect__max_df': [0.5],
        'vect__ngram_range': [(1, 1)],
        'clf__estimator__leaf_size': [10], #, 30, 50],
        'clf__estimator__n_neighbors': [10], #[5, 10, 20],
    }

    cv = GridSearchCV(pipeline, parameters, verbose=2)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function tests the model and reports f1 score, precision and 
    recall for each output category of the dataset.
    params:
    model: machine learning model pipeline 
    X_test: feature variables in test data
    Y_test: target variables in test data
    category_names: category names
    """
    Y_pred = model.predict(X_test)
    
    print("\nBest Parameters:", model.best_params_)
    
    for i, c in enumerate(category_names): 
        print("category: ", c) 
        print(classification_report(Y_test[i], Y_pred[i]))


def save_model(model, model_filepath):
    """
    This function exports the model as a pickle file.
    params:
    model: machine learning model to be saved
    model_filepath: pickle file name
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, table_name = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath, table_name)
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
