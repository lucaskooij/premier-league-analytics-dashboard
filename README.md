# Premier League Match Prediction & Analytics Dashboard

## Overview

This project explores whether historical Premier League match statistics can be used to predict match outcomes. Using Python, machine learning models, linear algebra concepts, and Power BI, the project analyzes match performance data and transforms it into interactive visual insights.

The goal was to investigate relationships between match statistics and results while demonstrating a complete analytics workflow from data preparation and feature engineering to predictive modeling and dashboard development.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Power BI
* GitHub

---

## Project Objectives

* Analyze historical Premier League match data
* Engineer meaningful performance features
* Predict match outcomes using machine learning models
* Apply linear algebra concepts to sports analytics
* Visualize results through an interactive Power BI dashboard
* Evaluate model performance and feature importance

---

## Data Pipeline

1. Imported historical Premier League match data.
2. Cleaned and standardized match statistics.
3. Created engineered features from halftime and full-time performance metrics.
4. Trained multiple prediction models.
5. Evaluated model performance using classification metrics.
6. Generated Power BI-ready datasets.
7. Built an interactive dashboard for exploration and analysis.

---

## Match Statistics Analyzed

The project utilizes several match-level features, including:

* Goals Scored
* Shots
* Shots on Target
* Corners
* Fouls
* Yellow Cards
* Red Cards

Additional engineered features include:

* Goal Differential
* Halftime Performance
* Second Half Performance
* Shot Differential
* Corner Differential
* Comeback Wins
* Blown Leads

---

## Machine Learning Models

The project evaluates multiple predictive approaches:

### Logistic Regression

Used to classify match outcomes based on historical match statistics and engineered features.

### Random Forest Classifier

Used to identify non-linear relationships within the data and generate feature importance rankings.

### Linear Algebra Prediction Model

A custom prediction framework based on:

**Aw = b**

Where:

* A = Match statistics matrix
* w = Feature weights
* b = Match outcomes

This model demonstrates how linear algebra concepts can be applied to predictive analytics.

---

## Similarity Analysis

The project implements Euclidean Distance (L2 Norm) to identify statistically similar matches.

This approach allows:

* Match comparison
* Pattern identification
* Historical similarity analysis

The assumption is that matches with similar statistical profiles may exhibit similar outcomes.

---

## Model Evaluation

Model performance was evaluated using:

* Accuracy Score
* Classification Reports
* Confusion Matrices
* Feature Importance Analysis

These metrics help assess predictive effectiveness and identify influential match statistics.

---

## Power BI Dashboard

The Power BI dashboard includes:

* Team Performance Analysis
* Match Outcome Distribution
* Home vs Away Performance
* Goal Scoring Trends
* Feature Importance Visualization
* Prediction Accuracy Tracking
* Interactive Filters and KPIs

The dashboard allows users to explore trends dynamically and gain insights from historical match data.

---

## Skills Demonstrated

* Python Programming
* Data Cleaning
* Feature Engineering
* Data Analysis
* Machine Learning
* Logistic Regression
* Random Forests
* Linear Algebra Applications
* Data Visualization
* Power BI Development
* Business Intelligence
* Sports Analytics

---

## Future Improvements

Potential enhancements include:

* XGBoost Models
* Real-Time Match Data Integration
* Expected Goals (xG) Metrics
* Team Form Tracking
* Hyperparameter Optimization
* Automated Dashboard Refreshes
* Advanced Prediction Models

---

## Dashboard Preview

*Screenshots of the Power BI dashboard will be added here.*

![Dashboard Screenshot](images/dashboard1.png)
