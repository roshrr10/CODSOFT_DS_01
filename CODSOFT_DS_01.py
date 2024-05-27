#!/usr/bin/env python
# coding: utf-8

# # TASK 1

# AIM: To build a predictive model that determines whether a passenger on the Titanic survived or not based on various features such as age, sex, passenger class, fare, and others. Additionally, it includes data preprocessing steps, visualizations for data exploration, and evaluation of the model's performance.

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[4]:


# Load the dataset
file_path = 'C:\\Users\\rosha\\Downloads\\Titanic-Dataset.csv'
data = pd.read_csv(file_path)


# In[5]:


# Data Preprocessing
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)


# In[6]:


# Convert 'Sex' to numerical values
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})


# In[7]:


# Fill missing values in 'Embarked' with the most common value
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# In[8]:


# Convert 'Embarked' to numerical values
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# In[9]:


# Fill missing values in 'Age' and 'Fare' with the median value
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)


# In[11]:


# Visualize survival rate by sex
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='Survived', data=data, palette='Set1')
plt.title('Survival Rate by Sex')
plt.show()


# In[12]:


# Visualize survival rate by class
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=data, palette='Set2')
plt.title('Survival Rate by Class')
plt.show()


# In[13]:


# Visualize age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution of Passengers')
plt.show()


# In[15]:


# Visualize fare distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Fare'], bins=30, kde=True, color='green')
plt.title('Fare Distribution of Passengers')
plt.show()


# In[16]:


# Select the relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']


# In[17]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[19]:


# Make predictions
y_pred = model.predict(X_test)


# In[20]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# In[21]:


print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)


# In[22]:


# Visualize the confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ### Inference

# - Data Imbalance: Visualizations such as count plots for survival rates by sex and class reveal imbalances in the dataset. For instance, more females survived compared to males.
# 
# - Age and Fare Distributions: Histograms show the distribution of age and fare among passengers, providing insights into the data spread.
# 
# - Model Performance: The logistic regression model's accuracy, classification report, and confusion matrix give a clear picture of its performance. The accuracy score indicates how well the model is predicting survival. The classification report provides precision, recall, and F1 scores, while the confusion matrix shows the number of true positives, true negatives, false positives, and false negatives.
# 
# These insights can help in understanding the factors affecting survival and the effectiveness of the predictive model.
