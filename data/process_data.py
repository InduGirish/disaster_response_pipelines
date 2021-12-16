import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads messages dataset and categories dataset from
    csv files and merges them using the common id.
    params:
    messages_filepath: messages dataset filepath
    categories_filepath: categories dataset filepath
    Returns merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    This function splits the 'catogories' column into separate category
    columns and also removes unwanted and duplicate data.
    params:
    df: input dataframe
    Returns cleaned dataframe
    """
    
    categories = df['categories'].str.split(';', expand=True)
    
    # use first row to extract a list of new column names for categories
    categories.columns = [x[0] for x in categories.iloc[0].str.split('-', n=1)]
    
    # Split categories into separate category columns
    for column in categories:
        categories[column] = [x[1] for x in categories[column] \
                                   .str.split('-', n=1)]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Remove labels 2 from category 'related'
    df = df[~(df['related']==2)]
    
    return df
    
def save_data(df, database_filename, table_name):
    """
    This function saves the dataset into an sqlite database.
    params:
    df: dataset to be saved 
    database_filename: sqlite database filename
    table_name: sqlite database  table name
    """
    engine = create_engine('sqlite:///' + database_filename)
    engine.execute(f"DROP TABLE IF EXISTS {table_name}")
    df.to_sql(table_name, engine, index=False)  


def main():
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, \
                           database_filepath, table_name = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db DisasterResponseTable' )


if __name__ == '__main__':
    main()