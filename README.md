# Disaster Response Pipeline
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

### Poject Motivation:
This project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. 
The initial dataset contains pre-labelled tweet and messages from real-life disaster. The goal is to classify emergency calls into categories: the analysis is performed using text-processing techniques to extract features from text data.

The Project is divided in the following Sections:

- Data Processing/ETL Pipeline: extract data from source, clean data and save them in a proper database structure
- Machine Learning Pipeline: train a model to classify text messages into 36 categories
- Web App: show model results in real time

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

