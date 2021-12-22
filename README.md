# Disaster Response Pipeline Project

The objective of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The dataset contains real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that we can send the messages to an appropriate disaster relief agency. 
The project includes a web app where an emergency worker can input a new message and get classification results in several categories and visualize it.


### File Structure:
- app
- | - template
- | |- master.html            # main page of web app
- | |- go.html                # classification result page of web app
- |- run.py                   # Flask file that runs app

- data
- |- disaster_categories.csv  # data to process 
- |- disaster_messages.csv    # data to process
- |- process_data.py          # ETL pipeline
- |- DisasterResponse.db      # database to save clean data to

- models
- |- train_classifier.py      # machine learning pipeline
- |- classifier.pkl           # saved model 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db DisasterResponseTable`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl DisasterResponseTable`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Result:

 ![Screenshots of Flask web app](https://github.com/InduGirish/disaster_responce_pipelines/blob/main/images/master.png)
 ![Model result](https://github.com/InduGirish/disaster_responce_pipelines/blob/main/images/result.png)
