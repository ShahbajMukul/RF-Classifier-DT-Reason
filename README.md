# RF-Classifier-DT-Reason

## Project Overview

This project is a machine learning model that predicts the Group (machine) and Code (reason) of a downtime event (Z4) based on various features such as machine type, machine error code, time of day, and machine location.

## Context

In a manufacturing plant, downtime events are recorded in a database. Each completed downtime event has a Group and Code associated with it. The Group represents the machine that caused the downtime, while the Code represents the reason for the downtime.

## Data Preparation

### Data

The dataset includes the following features:
- Machine Type
- Error Code
- Time of Day
- Machine Location
- Group (target variable)
- Code (target variable)

### Data Preprocessing

The data preprocessing steps include:
- One-hot encoding for categorical features
- Train-test split
- Standardization

## Model

The model used is a Random Forest Classifier.

## Evaluation

The model is evaluated using the following metrics:
- Accuracy
- Confusion Matrix

## Deployment

A simple Python application is developed for demonstration purposes. The model is trained and saved, and as the user inputs the features, the model predicts the Group and Code of the downtime event.

## Required Libraries

- pandas
- matplotlib
- seaborn
- scikit-learn
- graphviz
- plotly (optional, for the deployed app)

## How to Run

1. **Install the required libraries:**
    ```sh
    pip install pandas matplotlib seaborn scikit-learn graphviz plotly
    ```

2. **Generate the dataset:**
    ```python
    import pandas as pd
    import numpy as np
    import random

    # Generate the sample dataset as shown in the notebook
    ```

3. **Preprocess the data:**
    ```python
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import joblib

    # One-hot encode categorical variables
    # Standardize numerical features
    ```

4. **Train the model:**
    ```python
    from sklearn.ensemble import RandomForestClassifier

    # Split the data into training and testing sets
    # Train the Random Forest Classifier
    ```

5. **Evaluate the model:**
    ```python
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Evaluate the model on the testing set
    ```

6. **Deploy the model (optional):**
    ```python
    import joblib

    # Save the trained model
    # Develop a simple application to use the model for predictions
    ```

## Notebook

The Jupyter Notebook `Z4Classifier.ipynb` contains all the necessary code to generate the dataset, preprocess the data, train the model, evaluate the model, and visualize the results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
