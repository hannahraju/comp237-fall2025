'''

NAME: Hannah Raju
ID: 301543568
DATE: October 12, 2025
INFO: COMP 237 Assignment 3 - Linear Regression - EXERCISE 2

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# LOAD DATA
ecom_exp_hannah = pd.read_csv('Ecom Expense.csv')


# DATA EXPLORATION

## print first three records
print("The first three records:")
print(ecom_exp_hannah.head(3))

## print shape of dataframe
print("\nDataframe shape:")
print(ecom_exp_hannah.shape)

## print column names
print("\nColumn names:")
print(ecom_exp_hannah.columns)

## print types of columns
print("\nColumn datatypes:")
print(ecom_exp_hannah.dtypes)

## print column name and number missing values per column
print("\nNumber of missing values by column:")
print(ecom_exp_hannah.isnull().sum())


#DATA TRANSFORMATION
## encode categorical data (Gender, City Tier)
gender_encoded = pd.get_dummies(ecom_exp_hannah[['Gender']]).astype(int)
city_encoded = pd.get_dummies(ecom_exp_hannah[['City Tier']]).astype(int)

## add encoded gender columns to dataframe
ecom_exp_hannah = ecom_exp_hannah.join(gender_encoded)

## add encoded city tier columns to dataframe 
ecom_exp_hannah = ecom_exp_hannah.join(city_encoded)

## drop Gender, City Tier, Transaction ID from dataframe
ecom_exp_hannah.drop(columns=['Gender', 'City Tier', 'Transaction ID'], inplace=True)

## function to normalize dataframe
def normalize(df):

    df = df.apply(lambda x: (x-x.min())/(x.max()-x.min()))
    return df

## call function to normalize dataframe
ecom_exp_hannah = normalize(ecom_exp_hannah)

## print first two records
print("\nFirst two records of normalized dataset:")
print(ecom_exp_hannah.head(2))

## generate histogram for each column
plt.figure()
hist = ecom_exp_hannah.hist(figsize=(9,10))
plt.savefig('histogram.png')

## generate scattermatrix showing relationship between Age, Income, Transcastion Time, Total Spend
cols = ['Age ', 'Monthly Income','Transaction Time', 'Total Spend']
scatter_data = ecom_exp_hannah[cols]
plt.figure()
pd.plotting.scatter_matrix(scatter_data, alpha=0.4, figsize=(13,15))
plt.savefig('scattermatrix.png')

# BUILD MODEL (without Record)
print("\nMODEL: without 'Record")

## split isolate feature columns and split data into 65% train, 35% test
x_data = ecom_exp_hannah.drop(columns=['Age ', ' Items ', 'Record'])
y_data = ecom_exp_hannah['Total Spend']
x_train_hannah, x_test_hannah, y_train_hannah, y_test_hannah = train_test_split(x_data, y_data, train_size=0.65, test_size=0.35, random_state=68)

## fit linear regression model to the training data
reg = LinearRegression().fit(x_train_hannah, y_train_hannah)

## print model coefficients (weights) 
print("\nModel coefficients:")
print(reg.coef_)

## print model score (R2)
print("\nModel score:")
print(reg.score(x_test_hannah, y_test_hannah))

# BUILD MODEL (with Record)
print("\nMODEL: with 'Record'")
x_data = ecom_exp_hannah.drop(columns=['Age ', ' Items ',])
y_data = ecom_exp_hannah['Total Spend']
x_train_hannah, x_test_hannah, y_train_hannah, y_test_hannah = train_test_split(x_data, y_data, train_size=0.65, test_size=0.35, random_state=68)

reg = LinearRegression().fit(x_train_hannah, y_train_hannah)
print("\nModel coefficients:")
print(reg.coef_)
print("\nModel score:")
print(reg.score(x_test_hannah, y_test_hannah))
