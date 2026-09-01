# 🍔 Food Delivery Time Prediction Using Machine Learning

## 📌 Overview

This project focuses on predicting food delivery time using machine learning techniques. The system analyzes factors such as delivery distance, traffic conditions, weather, time of day, vehicle type, restaurant preparation time, and delivery-person experience to estimate the expected delivery duration.

The project compares machine learning models against a simple baseline to determine whether machine learning can provide more accurate delivery-time predictions.

---

## 🎯 Research Question

**Can machine learning accurately predict food delivery time using order, restaurant, delivery, traffic, and weather-related features?**

---

## 🎯 Objectives

- Analyze factors that influence food delivery time.
- Clean and preprocess the dataset.
- Identify important features affecting delivery duration.
- Build a baseline prediction method.
- Train machine learning regression models.
- Compare model performance with the baseline.
- Evaluate models using MAE, RMSE, and R².
- Generate practical recommendations for improving delivery-time estimation.
- Make the research work reproducible.

---

## 📊 Dataset

The dataset contains information related to food delivery orders.

### Example Features

| Feature | Description |
|---|---|
| Distance_km | Distance between restaurant and customer |
| Weather | Weather condition during delivery |
| Traffic_Level | Traffic condition |
| Time_of_Day | Morning, afternoon, evening, or night |
| Vehicle_Type | Type of vehicle used |
| Restaurant_Preparation_Time | Time taken by restaurant to prepare order |
| Delivery_Person_Experience | Experience of delivery person |
| Order_Size | Number of items in the order |
| Delivery_Time | Actual delivery time in minutes |

### Target Variable

**Delivery_Time**

The target represents the actual time taken to deliver an order, measured in minutes.

> Dataset source and data-credit information will be documented here based on the dataset used in the experiment.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Baseline Model
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Results & Visualization
   ↓
Ranked Recommendations
