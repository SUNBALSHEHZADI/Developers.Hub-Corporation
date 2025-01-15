# Developers.Hub-Corporation
Data Science Internship

# Data Science Intern Tasks

This repository contains solutions for four data science tasks, each focused on different aspects of the data science workflow. Below is a detailed explanation of each task, the steps involved, and the expected outcomes. The tasks include Exploratory Data Analysis (EDA), Sentiment Analysis, Fraud Detection, and Predicting House Prices.

## Table of Contents

- [Data Science Intern Tasks](#data-science-intern-tasks)
  - [Table of Contents](#table-of-contents)
  - [Task 1: EDA and Visualization of a Real-World Dataset](#task-1-eda-and-visualization-of-a-real-world-dataset)
  - [Task 2: Text Sentiment Analysis](#task-2-text-sentiment-analysis)
  - [Task 3: Fraud Detection System](#task-3-fraud-detection-system)
  - [Task 4: Predicting House Prices Using the Boston Housing Dataset](#task-4-predicting-house-prices-using-the-boston-housing-dataset)
  - [Running the Scripts](#running-the-scripts)
  - [Observations and Insights](#observations-and-insights)
    - [Task 1: EDA and Visualization of a Real-World Dataset](#task-1-eda-and-visualization-of-a-real-world-dataset-1)
    - [Task 2: Text Sentiment Analysis](#task-2-text-sentiment-analysis-1)
    - [Task 3: Fraud Detection System](#task-3-fraud-detection-system-1)
    - [Task 4: Predicting House Prices](#task-4-predicting-house-prices)
    - [Conclusion](#conclusion)

---

## Task 1: EDA and Visualization of a Real-World Dataset

**Objective:**  
Perform an Exploratory Data Analysis (EDA) on the Airbnb Listings Dataset. The goal is to understand the data, clean it, and provide insights through visualizations.

**Steps:**
1. **Load the Dataset**: Use Pandas to load and explore the dataset.
2. **Data Cleaning**:
   - Handle missing values using imputation techniques or removal.
   - Remove duplicate rows.
   - Identify and manage outliers using statistical methods or visualizations.
3. **Visualizations**:
   - Create bar charts for categorical variables.
   - Plot histograms for numeric distributions.
   - Generate a correlation heatmap for numeric features.


---

## Task 2: Text Sentiment Analysis

**Objective:**  
Build a sentiment analysis model using a dataset such as IMDB Reviews to predict sentiment (positive or negative) based on text input.

**Steps:**
1. **Text Preprocessing**:
   - Tokenize the text into individual words.
   - Remove stopwords.
   - Perform lemmatization for text normalization.
2. **Feature Engineering**:
   - Convert text data into numerical format using TF-IDF or word embeddings.
3. **Model Training**:
   - Train a classifier (e.g., Logistic Regression) to predict sentiment.
4. **Model Evaluation**:
   - Evaluate the model's performance using metrics like precision, recall, and F1-score.


---

## Task 3: Fraud Detection System

**Objective:**  
Develop a fraud detection system using a dataset like the Credit Card Fraud Dataset to classify transactions as either fraudulent or legitimate.

**Steps:**
1. **Data Preprocessing**:
   - Handle imbalanced data using techniques like SMOTE (Synthetic Minority Over-sampling Technique).
2. **Model Training**:
   - Train a Random Forest model to detect fraudulent transactions.
3. **Model Evaluation**:
   - Evaluate the system’s precision, recall, and F1-score.
4. **Testing Interface**:
   - Create a simple interface (e.g., a command-line input) to test the fraud detection system.

---

## Task 4: Predicting House Prices Using the Boston Housing Dataset

**Objective:**  
Build a regression model from scratch to predict house prices using the Boston Housing Dataset.

**Steps:**
1. **Data Preprocessing**:
   - Normalize numerical features and preprocess categorical variables.
2. **Model Implementation**:
   - Implement Linear Regression, Random Forest, and XGBoost models from scratch (without using built-in libraries like `sklearn.linear_model`).
3. **Performance Comparison**:
   - Compare the models using metrics such as RMSE (Root Mean Squared Error) and R² (Coefficient of Determination).
4. **Feature Importance**:
   - Visualize feature importance for tree-based models.


---

## Running the Scripts

To run the scripts, follow these steps:

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/SUNBALSHEHZADI/data-science-intern-tasks.git
    cd data-science-intern-tasks
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the individual scripts for each task:
    - Task 1 (EDA and Visualization): `python task1_eda.py`
    - Task 2 (Sentiment Analysis): `python task2_sentiment_analysis.py`
    - Task 3 (Fraud Detection): `python task3_fraud_detection.py`
    - Task 4 (House Price Prediction): `python task4_house_price_prediction.py`

4. Follow the instructions in the script to input data and view results.

---

## Observations and Insights

### Task 1: EDA and Visualization of a Real-World Dataset
- The analysis revealed that certain columns had a significant amount of missing data, such as price and location-related fields.
- Most listings were clustered in urban areas, and there were noticeable outliers in pricing (e.g., ultra-expensive properties).
- A correlation heatmap showed that price had a strong positive correlation with features like number of bedrooms and availability.

### Task 2: Text Sentiment Analysis
- The preprocessing steps (tokenization, stopword removal, and lemmatization) improved model performance by reducing noise in the text.
- The Logistic Regression model achieved a good balance between precision and recall, with an F1-score indicating solid performance.
- TF-IDF was effective in representing text data for classification tasks.

### Task 3: Fraud Detection System
- SMOTE helped to balance the dataset by generating synthetic samples for the minority class (fraudulent transactions).
- The Random Forest model performed well in detecting fraud, with precision and recall metrics providing insights into false positives and false negatives.
- The testing interface allowed for real-time detection of fraudulent transactions.

### Task 4: Predicting House Prices
- The custom implementations of Linear Regression, Random Forest, and XGBoost provided a comparison of the models' performance.
- Random Forest and XGBoost outperformed Linear Regression in terms of both RMSE and R².
- Feature importance visualizations helped identify which factors (e.g., number of rooms, crime rate) were most influential in predicting house prices.

---

### Conclusion

This repository showcases four essential tasks in the data science workflow, covering data analysis, text processing, model development, and performance evaluation. By following the steps outlined in each task, data science interns can gain hands-on experience in solving real-world problems using Python and machine learning techniques.

---
