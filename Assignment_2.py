import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


st.title('Assignment 2')


#1a. Assign target variable (success) with values 0 (unsuccessful) and 1 (successful)

def load_data():
    # Use the full file path to access the file
    file_path = '/Users/pascasugijarto/Documents/TIM5301/kickstarter_2016.csv'
    df = pd.read_csv(file_path) 
    
    # Create a target variable 'success'
    df['success'] = df['State'].apply(lambda x: 1 if x.lower() == 'successful' else 0)
    
    return df

# Load the data and check the first few rows
df = load_data()
print(df[['State', 'success']].head(10))


# 1. Create two additional features: the duration of the campaign in days, and the length of the project name in words
# Show the new features: 'Duration_Days' and 'Project_Name_Length_Words'

def load_data():
    # Use the full file path to access the file
    file_path = '/Users/pascasugijarto/Documents/TIM5301/kickstarter_2016.csv'
    df = pd.read_csv(file_path)
    
    # Create a new column for the log of the funding goal, replacing 0 with NaN to avoid log(0) error
    df['log_goal'] = np.log(df['Goal'].replace(0, np.nan))
    
    # Create a new column for the log of the backers, replacing 0 with NaN to avoid log(0) error
    df['log_backers'] = np.log(df['Backers'].replace(0, np.nan))
    
    # Convert 'Launched' and 'Deadline' to datetime format
    df['Launched'] = pd.to_datetime(df['Launched'])
    df['Deadline'] = pd.to_datetime(df['Deadline'])
    
    # Create a new column for the campaign duration in days
    df['duration'] = (df['Deadline'] - df['Launched']).dt.days
    
    # Create a new column for the length of the project name in words
    df['name_length'] = df['Name'].apply(lambda x: len(x.split()))
    
    # Return the first 10 records with the new features
    return df[['Goal', 'log_goal', 'Backers', 'log_backers', 'duration', 'name_length']].head(10)

# Call the function and print the output
df_top_10 = load_data()
print(df_top_10)

#3. Model creation and evaluation

# Count the number of success and fail campaigns
success_count = df['success'].sum()
fail_count = len(df) - success_count

print(f"Number of successful campaigns: {success_count}")
print(f"Number of failed campaigns: {fail_count}")

# File path (if running locally)
file_path = '/Users/pascasugijarto/Documents/TIM5301/kickstarter_2016.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Ensure 'Launched' and 'Deadline' columns are datetime types
df['Launched'] = pd.to_datetime(df['Launched'])
df['Deadline'] = pd.to_datetime(df['Deadline'])

# Encode success (1 for 'successful', 0 for others) directly in the dataframe
df['success'] = df['State'].apply(lambda x: 1 if x.strip().lower() == 'successful' else 0)

# Manually calculate the features directly in the dataframe
df['log_goal'] = np.log1p(df['Goal'])  # Log transformation of 'Goal'
df['duration'] = (df['Deadline'] - df['Launched']).dt.days  # Calculate campaign duration in days
df['name_length'] = df['Name'].apply(len)  # Calculate name length

# Sidebar for user inputs
st.sidebar.title("Model & Feature Selection")

# Classifier selection
classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "Random Forest", "Gradient Boosting"))

# Feature selection
selected_features = st.sidebar.multiselect("Select Features", ["log_goal", "duration", "name_length"], default=["log_goal", "duration", "name_length"])

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(max_samples=0.1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Function to create the full pipeline with selected classifier
def create_pipeline(selected_classifier, selected_features):
    classifier = classifiers.get(selected_classifier, LogisticRegression())
    
    # Preprocessing pipeline to select only the chosen features
    pipeline = Pipeline([
        ('select_features', FunctionTransformer(lambda x: x[selected_features], validate=False)),
        ('classifier', classifier)
    ])
    
    return pipeline

# Prepare data for model training
X = df[selected_features]  # Features chosen by the user
y = df['success']  # Target (success)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to perform cross-validation and evaluate the model
def cross_validate_and_evaluate(selected_classifier, selected_features, k=5):
    pipeline = create_pipeline(selected_classifier, selected_features)
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary')
    }
    
    # Perform cross-validation on the training data
    cv_scores = cross_validate(pipeline, X_train, y_train, cv=k, scoring=scoring)
    
    # Extract and calculate mean cross-validation scores for training data
    mean_train_accuracy = np.mean(cv_scores['test_accuracy']) * 100
    mean_train_precision = np.mean(cv_scores['test_precision']) * 100
    mean_train_recall = np.mean(cv_scores['test_recall']) * 100
    mean_train_f1 = np.mean(cv_scores['test_f1']) * 100
    
    # Train the model on the full training data
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    test_accuracy = accuracy_score(y_test, pipeline.predict(X_test)) * 100
    test_precision = precision_score(y_test, pipeline.predict(X_test)) * 100
    test_recall = recall_score(y_test, pipeline.predict(X_test)) * 100
    test_f1 = f1_score(y_test, pipeline.predict(X_test)) * 100
    
    return (mean_train_accuracy, mean_train_precision, mean_train_recall, mean_train_f1), (test_accuracy, test_precision, test_recall, test_f1)

# Display results when user selects "Run"
if st.sidebar.button("Run Model"):
    # Run cross-validation and get the mean scores
    train_scores, test_scores = cross_validate_and_evaluate(classifier_name, selected_features)
    
    # Display the training set results
    st.write(f"### Training Results for {classifier_name}")
    st.write(f"**Mean Accuracy** (Train): {train_scores[0]:.2f}%")
    st.write(f"**Mean Precision** (Train): {train_scores[1]:.2f}%")
    st.write(f"**Mean Recall** (Train): {train_scores[2]:.2f}%")
    st.write(f"**Mean F1 Score** (Train): {train_scores[3]:.2f}%")
    
    # Display the test set results
    st.write(f"### Test Results for {classifier_name}")
    st.write(f"**Accuracy** (Test): {test_scores[0]:.2f}%")
    st.write(f"**Precision** (Test): {test_scores[1]:.2f}%")
    st.write(f"**Recall** (Test): {test_scores[2]:.2f}%")
    st.write(f"**F1 Score** (Test): {test_scores[3]:.2f}%")










