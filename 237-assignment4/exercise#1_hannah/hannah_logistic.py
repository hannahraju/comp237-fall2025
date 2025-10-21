'''
NAME: Hannah Raju
ID: 301543568
DATE: October 23, 2025
INFO: COMP 237: Assignment 4 - Logistic Regression
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# a. Get the data
titanic_hannah = pd.read_csv("titanic.csv")

# b. Initial exploration

print(titanic_hannah.head(3)) # display first three records
print(titanic_hannah.shape) # display shape of dataframe
print(titanic_hannah.info()) # display names, datatypes, and counts of each column
print(titanic_hannah.isnull().sum()) # calculate number of missing values per column
print(titanic_hannah['Sex'].unique()) # display each unique value in Sex
print(titanic_hannah['Pclass'].unique()) # display each unique value in Pclass

# c. Data visualization

## bar chart Passenger Class vs. Survived
ct_class = pd.crosstab(titanic_hannah['Survived'], titanic_hannah['Pclass']) # create cross tabulation of Survived x Passenger Class
plt.figure() # define new figure
ct_class.loc[1].plot(kind='bar') #plot from crosstab's index 1 (Survival = True)
plt.xlabel("Passenger Class") #label x axis
plt.ylabel("Number Survived") #label y axis
plt.title("Titanic: Number of Passengers Survived vs. Passenger Class (Hannah Raju)") #title plot
plt.savefig("bar1_hannah.png") #save figure to file

## bar chart Passenger Sex vs Survived
ct_sex = pd.crosstab(titanic_hannah['Survived'], titanic_hannah['Sex']) # create cross tabulation Survival vs Passenger sex
plt.figure() #define new figure
ct_sex.loc[1].plot(kind='bar', figsize=(6,8)) #plot from crosstab's index 1 (survival = true)
plt.xlabel("Passenger Sex") #label x axis
plt.ylabel("Number Survived") #label y axis
plt.title("Titanic: Number of Passengers Survived vs. Sex (Hannah Raju)") #title plot
plt.savefig("bar2_hannah.png") #save figure to file "bar2_hannah.png"

## make scatter matrix - this isn't the right application for a scattermatrix

titanic_hannah_survived = titanic_hannah[titanic_hannah['Survived']==1] #isolate data where survival=true
print(titanic_hannah_survived)
cols = ['Sex', 'Pclass', 'Fare', 'SibSp', 'Parch'] # labels of columns to compare
scatter_data = titanic_hannah_survived[cols] #define scatter data as dataframe columns declared above
plt.figure() #define new figure
pd.plotting.scatter_matrix(scatter_data) #create scatter matrix
plt.savefig("scatter.png") #save figure to file "scatter.png"

'''
## make bar charts plots for each
fig, axs = plt.subplots(2,3, figsize = (15,10)) #declare new subplots

### plot Survived vs. Passenger Sex
axs[0,0].scatter(ct_sex.columns, ct_sex.loc[1])
axs[0,0].set_xlabel("Passenger sex (Male/Female)")
axs[0,0].set_ylabel("Number of passengers survived")

### plot Survived vs. Passenger class 
axs[0,1].scatter(ct_class.columns, ct_class.loc[1])
axs[0,1].set_xlabel("Passenger class")
axs[0,1].set_ylabel("Number of passengers survived")

### plot Survived vs. Passenger fare
ct_fare = pd.crosstab(titanic_hannah['Survived'], titanic_hannah['Fare'])
axs[0,2].scatter(ct_fare.columns,ct_fare.loc[1])
axs[0,2].set_xlabel("Passenger fare ($)")
axs[0,2].set_ylabel("Number of passengers survived")

### plot Survived vs. Passenger Siblings/Spouses
ct_sibsp = pd.crosstab(titanic_hannah['Survived'], titanic_hannah['SibSp'])
axs[1,0].scatter(ct_sibsp.columns, ct_sibsp.loc[1])
axs[1,0].set_xlabel("# Passenger Siblings/Spouses")
axs[1,0].set_ylabel("Number of passengers survived")

### plot Survived cs. Passenger Parents/Children
ct_parch = pd.crosstab(titanic_hannah["Survived"], titanic_hannah['Parch'])
axs[1,1].scatter(ct_parch.columns, ct_parch.loc[1])
axs[1,1].set_xlabel("# Passenger parents/children")
axs[1,1].set_ylabel("Number of passengers survived")

axs[-1, -1].axis('off') #remove unused figured
plt.savefig("scatter_2.png") #save figure to "scatter_2.png"
'''

# d. Data Transformation

## drop unwanted columns
titanic_hannah.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True)

## encode categorical data
sex_encoded = pd.get_dummies(titanic_hannah[['Sex']]).astype(int)
embarked_encoded = pd.get_dummies(titanic_hannah[['Embarked']]).astype(int)

## drop og columns and add encoded ones to df
titanic_hannah.drop(columns=['Sex', 'Embarked'], inplace=True)
titanic_hannah = titanic_hannah.join(sex_encoded)
titanic_hannah = titanic_hannah.join(embarked_encoded)

## replace missing age values with mean age
titanic_hannah['Age'] = titanic_hannah['Age'].fillna(titanic_hannah['Age'].mean())

## change all columns to type float
titanic_hannah = titanic_hannah.astype("float")

print(titanic_hannah.info())

## write function to normalize dataframe
def normalize(df):
    df = df.apply(lambda x: (x-x.min())/(x.max()-x.min()))
    return df


## normalize dataframe and display first 2 records
titanic_hannah_normalized = normalize(titanic_hannah)
print(titanic_hannah_normalized.head(2))

## histogram for each variable
plt.figure()
titanic_hannah_normalized.hist(figsize=(9,10))
plt.savefig("histogram.png")

## histogram for port of embarkation
titanic_hannah_normalized_survived = titanic_hannah_normalized[titanic_hannah_normalized['Survived'] == 1.0].drop(columns='Survived')
plt.figure()
bins=[0,1]
x1 = titanic_hannah_normalized_survived[titanic_hannah_normalized_survived['Embarked_C'] == 1.0]['Embarked_C']
x2 = titanic_hannah_normalized_survived[titanic_hannah_normalized_survived['Embarked_Q'] == 1.0]['Embarked_Q']
x3 = titanic_hannah_normalized_survived[titanic_hannah_normalized_survived['Embarked_S'] == 1.0]['Embarked_S']
labels = ['Cherbourg', 'Queenstown','Southampton']
plt.hist([x1,x2,x3], bins, label=labels)
plt.legend(loc='upper left')
plt.xticks([])
plt.title("Titanic Passengers Survived by Port of Embarkation")
plt.ylabel("Number Survived")
plt.xlabel("Port of Embarkation")
plt.savefig("histogram_embarked.png")


# split the data with sklearn
x_hannah = titanic_hannah_normalized.drop(columns=['Survived'])
y_hannah = titanic_hannah_normalized['Survived']
x_train_hannah, x_test_hannah, y_train_hannah, y_test_hannah = train_test_split(x_hannah, y_hannah, train_size=0.7, test_size=0.3, random_state=68)

# e. Build and validate the model
hannah_model = LogisticRegression(solver='liblinear', random_state=68)
hannah_model.fit(x_train_hannah, y_train_hannah)

## display coefficients
print(pd.DataFrame(zip(x_train_hannah.columns, np.transpose(hannah_model.coef_))))

## cross validation from test_size 10% to 50%
for i in np.arange(0.10, 0.55, 0.05):
    x_train_hannah, x_test_hannah, y_train_hannah, y_test_hannah = train_test_split(x_hannah, y_hannah, train_size = 1-i, test_size=i, random_state=68)
    hannah_model = LogisticRegression(solver='liblinear', random_state=68)
    hannah_model.fit(x_train_hannah, y_train_hannah)
    scores = cross_val_score(hannah_model, x_train_hannah, y_train_hannah, cv=10)
    print("Test size: " + f"{i:.2f}")
    print("min: " + f"{scores.min():.3f}")
    print("mean: " + f"{scores.mean():.3f}")
    print("max: " +f"{scores.max():.3f}")
    print()


# f. Retrain and Test the model
x_train_hannah, x_test_hannah, y_train_hannah, y_test_hannah = train_test_split(x_hannah, y_hannah, train_size = 0.7, test_size=0.3, random_state=68)
hannah_model = LogisticRegression(solver='liblinear', random_state=68)
hannah_model.fit(x_train_hannah, y_train_hannah)


y_pred_hannah = hannah_model.predict_proba(x_test_hannah)
y_pred_hannah_train = hannah_model.predict_proba(x_train_hannah)

## threshold = 0.5
print('THRESHOLD = 0.5')
y_pred_hannah_flag = y_pred_hannah[:,1] > 0.5
y_pred_hannah_flag_train = y_pred_hannah_train[:,1] > 0.5

test_accuracy = accuracy_score(y_test_hannah, y_pred_hannah_flag)
train_accuracy = accuracy_score(y_train_hannah, y_pred_hannah_flag_train)
print("Test Accuracy: " +str(test_accuracy))
print("Train Accuracy: "+str(train_accuracy))

cm = confusion_matrix(y_test_hannah, y_pred_hannah_flag)
print(cm)

print(classification_report(y_test_hannah, y_pred_hannah_flag))

## threshold = 0.75
print('THRESHOLD = 0.75')
y_pred_hannah_flag = y_pred_hannah[:,1] > 0.75
y_pred_hannah_flag_train = y_pred_hannah_train[:,1] > 0.75

test_accuracy = accuracy_score(y_test_hannah, y_pred_hannah_flag)
train_accuracy = accuracy_score(y_train_hannah, y_pred_hannah_flag_train)
print("Test Accuracy: " +str(test_accuracy))
print("Train Accuracy: "+str(train_accuracy))

cm = confusion_matrix(y_test_hannah, y_pred_hannah_flag)
print(cm)

print(classification_report(y_test_hannah, y_pred_hannah_flag))