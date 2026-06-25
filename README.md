# Netflix Content Segmentation Using Unsupervised Machine Learning

## Overview

This project applies unsupervised machine learning techniques to segment Netflix content into meaningful groups based on content characteristics. The objective is to identify hidden patterns within the dataset and provide insights into different categories of movies and TV shows.

The project includes data preprocessing, exploratory data analysis, clustering model development, cluster evaluation, visualization, and deployment through an interactive Streamlit application.

## Problem Statement

Streaming platforms generate large volumes of content-related data. Understanding natural groupings within this data can help improve content recommendations, audience targeting, and content strategy.

This project aims to discover these hidden patterns using clustering algorithms without predefined labels.

## Objectives

* Perform data cleaning and preprocessing.
* Conduct exploratory data analysis (EDA).
* Scale and prepare features for clustering.
* Apply clustering algorithms to identify content segments.
* Evaluate clustering performance using standard metrics.
* Visualize and interpret cluster characteristics.
* Build an automated machine learning pipeline.
* Deploy the solution using Streamlit.

## Dataset

The dataset contains Netflix content information, including numerical and categorical features describing movies and TV shows.

### Key Steps

#### Data Cleaning

* Handling missing values
* Removing duplicate records
* Feature selection
* Data transformation

#### Exploratory Data Analysis

* Statistical summaries
* Distribution analysis
* Correlation analysis
* Data visualization

#### Feature Engineering

* Numerical feature scaling
* Data preparation for clustering

## Clustering Techniques

### K-Means Clustering

Used to partition content into distinct groups based on similarity.

### Hierarchical Clustering

Applied to analyze cluster relationships and hierarchy.

### DBSCAN

Used to identify density-based clusters and potential outliers.

## Cluster Evaluation

The clustering models were evaluated using:

* Silhouette Score
* Davies-Bouldin Index
* Elbow Method

## Machine Learning Pipeline

A Scikit-Learn pipeline was created to automate:

1. Missing value handling
2. Feature scaling
3. Clustering model training
4. Cluster prediction

## Streamlit Application

The project includes an interactive Streamlit application that allows users to:

* Input content-related features
* Predict cluster assignments
* Visualize cluster distributions
* Explore cluster characteristics

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Plotly
* Joblib
* Streamlit

## Project Structure

```text
Netflix-Content-Segmentation/
│
├── data/
│   └── netflix_dataset.csv
│
├── notebooks/
│   └── Netflix_Content_Segmentation.ipynb
│
├── models/
│   ├── netflix_cluster_pipeline.pkl
│   └── netflix_features.pkl
│
├── app.py
├── train_pipeline.py
├── requirements.txt
└── README.md
```

## Results

* Successfully identified distinct Netflix content segments.
* Generated meaningful clusters based on content characteristics.
* Evaluated clustering quality using multiple performance metrics.
* Built a reusable clustering pipeline for future predictions.
* Developed an interactive Streamlit application for cluster analysis.

## Future Improvements

* Incorporate additional content metadata.
* Experiment with advanced clustering techniques.
* Implement recommendation-based cluster insights.
* Integrate real-time content analysis.
