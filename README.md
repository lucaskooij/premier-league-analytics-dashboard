# Premier League Analytics Dashboard

## Overview

This project analyzes historical Premier League match data using Python and Power BI to uncover patterns in team performance and match outcomes. The objective was to explore how match statistics such as goals, shots, corners, fouls, and disciplinary records relate to match results while building an interactive dashboard for data exploration and decision-making.

The project combines data cleaning, feature engineering, statistical analysis, and business intelligence visualization to transform raw football data into actionable insights.

---

## Project Objectives

* Analyze historical Premier League match data
* Explore relationships between match statistics and outcomes
* Apply concepts from data science, linear algebra, and optimization
* Develop an interactive Power BI dashboard for visual analysis
* Demonstrate a complete data analytics workflow from raw data to visualization

---

## Technologies Used

* Python
* Pandas
* NumPy
* Power BI
* GitHub

---

## Data Pipeline

1. Imported historical Premier League match data.
2. Cleaned and standardized statistical features.
3. Removed inconsistencies and handled missing values.
4. Prepared the dataset for analysis and visualization.
5. Generated analytical metrics and outcome indicators.
6. Exported processed data for Power BI integration.
7. Built an interactive dashboard to explore trends and performance metrics.

---

## Match Statistics Analyzed

The project focuses on several key match statistics:

* Goals Scored
* Shots
* Shots on Target
* Corners
* Fouls
* Yellow Cards
* Red Cards

These variables were used to evaluate team performance and investigate how statistical patterns relate to match outcomes.

---

## Analytical Approach

### Data Representation

Each match is represented as a numerical feature vector containing match statistics.

When combined, these vectors form a dataset where:

* Rows represent individual matches
* Columns represent statistical features

This structure allows the data to be analyzed using statistical and machine learning concepts.

### Similarity Analysis

One technique explored was Euclidean Distance (L2 Norm) to measure similarity between matches.

The underlying assumption is that matches with similar statistical profiles may exhibit similar outcomes.

### Prediction Framework

The project investigated how weighted statistical features can be used to model match outcomes conceptually.

This approach provides a foundation for future machine learning implementations while remaining interpretable and easy to understand.

---

## Dashboard Features

The Power BI dashboard includes:

* Team Performance Comparisons
* Home vs Away Analysis
* Match Outcome Distributions
* Scoring Trends
* Interactive Filters
* KPI Tracking
* Historical Performance Insights

The dashboard enables users to explore data dynamically and identify meaningful trends across teams and seasons.

---

## Skills Demonstrated

* Python Programming
* Data Cleaning
* Data Transformation
* Feature Engineering
* Data Analysis
* Data Visualization
* Power BI Development
* Business Intelligence
* Linear Algebra Applications
* Sports Analytics

---

## Limitations

While the project provides valuable analytical insights, several limitations exist:

* Primarily focuses on historical match statistics
* Does not account for injuries or player availability
* Does not include tactical or managerial factors
* External conditions are not considered
* Prediction concepts remain exploratory rather than production-ready

---

## Future Improvements

Potential enhancements include:

* Logistic Regression Models
* Random Forest Models
* XGBoost Integration
* Real-Time Data Integration
* Rolling Team Form Metrics
* Expected Goals (xG) Analysis
* Automated Dashboard Refreshes
* Advanced Predictive Modeling

---

## Dashboard Preview

*Power BI dashboard screenshots will be added here.*

![Dashboard Screenshot](images/dashboard1.png)
