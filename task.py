#testing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
titanic_data = pd.read_csv('titanic.csv')

# 1. Data Analysis & Preprocessing:
# Identify attribute types
# - Ordinal: Pclass
# - Nominal: Sex, Embarked
# - Binary: Survived
# - Discrete: SibSp, Parch
# - Continuous: Age, Fare

# Justify preprocessing steps:
# - Ordinal: No preprocessing needed
# - Nominal: Convert to numerical using one-hot encoding
# - Binary: No preprocessing needed
# - Discrete: No preprocessing needed
# - Continuous: Handle missing values (imputation), scale if necessary

# Preprocessing
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# Duplicate the dataset
titanic_data_unprocessed = titanic_data.copy()

titanic_data = titanic_data.drop(columns='Cabin', axis=1)

titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)



# Duplicate the dataset
titanic_data_unprocessed = titanic_data.copy()

# 2. Data Visualization:
# Basic count plots
sns.set()

sns.countplot(x='Survived', data=titanic_data)

sns.countplot(x='Sex', hue='Survived', data=titanic_data)

sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

# Additional visualizations
plt.figure(figsize=(10, 6))
sns.pairplot(titanic_data)
plt.show()

plt.figure(figsize=(10, 6))
sns.distplot(titanic_data['Age'])
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=titanic_data)
plt.title('Age Distribution by Survival')
plt.show()

# 3. Data Cleaning:
# Handle outliers
outliers = titanic_data[(titanic_data['Age'] > 70) | (titanic_data['Fare'] > 100)]
titanic_data = titanic_data.drop(outliers.index)

# 4. Data Transformation:
# Scale numerical features
numeric_cols = ['Age', 'Fare']
scaler = StandardScaler()
titanic_data[numeric_cols] = scaler.fit_transform(titanic_data[numeric_cols])

# 5. Classification:
# Split data into train and test sets
X_processed  = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y_processed  = titanic_data['Survived']
X_train_processed, X_test_processed, Y_train_processed, Y_test_processed  = train_test_split(X_processed, Y_processed , test_size=0.2, random_state=2)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine (SVM)': SVC(),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Define a function to evaluate the model
def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    # Evaluate on training set
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, train_predictions)
    train_confusion_matrix = confusion_matrix(Y_train, train_predictions)
    train_classification_report = classification_report(Y_train, train_predictions)

    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, test_predictions)
    test_confusion_matrix = confusion_matrix(Y_test, test_predictions)
    test_classification_report = classification_report(Y_test, test_predictions)

    # Display evaluation results
    print("Training Set:")
    print("Accuracy:", train_accuracy)
    print("Confusion Matrix:\n", train_confusion_matrix)
    print("Classification Report:\n", train_classification_report)

    print("\nTest Set:")
    print("Accuracy:", test_accuracy)
    print("Confusion Matrix:\n", test_confusion_matrix)
    print("Classification Report:\n", test_classification_report)

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_processed, Y_train_processed)
    print(f"Evaluation on {name}:")
    evaluate_model(model, X_train_processed, Y_train_processed, X_test_processed, Y_test_processed)
    print("\n\n")

# Train and evaluate models on unprocessed data
X_unprocessed = titanic_data_unprocessed.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y_unprocessed = titanic_data_unprocessed['Survived']
X_train_unprocessed, X_test_unprocessed, Y_train_unprocessed, Y_test_unprocessed = train_test_split(X_unprocessed, Y_unprocessed, test_size=0.2, random_state=2)

# Train and evaluate models on unprocessed data
for name, model in models.items():
    model.fit(X_train_unprocessed, Y_train_unprocessed)
    print(f"Evaluation on {name} (Unprocessed Data):")
    evaluate_model(model, X_train_unprocessed, Y_train_unprocessed, X_test_unprocessed, Y_test_unprocessed)
    print("\n\n")