import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class TestModels(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.titanic_data = pd.read_csv('titanic.csv')

        # Preprocess data
        self.titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
        self.titanic_data = self.titanic_data.drop(columns='Cabin', axis=1)
        self.titanic_data['Age'].fillna(self.titanic_data['Age'].mean(), inplace=True)
        self.titanic_data['Embarked'].fillna(self.titanic_data['Embarked'].mode()[0], inplace=True)
        outliers = self.titanic_data[(self.titanic_data['Age'] > 70) | (self.titanic_data['Fare'] > 100)]
        self.titanic_data = self.titanic_data.drop(outliers.index)
        numeric_cols = ['Age', 'Fare']
        scaler = StandardScaler()
        self.titanic_data[numeric_cols] = scaler.fit_transform(self.titanic_data[numeric_cols])

        # Split data
        X = self.titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
        Y = self.titanic_data['Survived']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    def test_logistic_regression(self):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train, self.Y_train)
        train_predictions = model.predict(self.X_train)
        test_predictions = model.predict(self.X_test)
        self.assertTrue(accuracy_score(self.Y_train, train_predictions) > 0)
        self.assertTrue(accuracy_score(self.Y_test, test_predictions) > 0)

    def test_naive_bayes(self):
        model = GaussianNB()
        model.fit(self.X_train, self.Y_train)
        train_predictions = model.predict(self.X_train)
        test_predictions = model.predict(self.X_test)
        self.assertTrue(accuracy_score(self.Y_train, train_predictions) > 0)
        self.assertTrue(accuracy_score(self.Y_test, test_predictions) > 0)

    def test_support_vector_machine(self):
        model = SVC()
        model.fit(self.X_train, self.Y_train)
        train_predictions = model.predict(self.X_train)
        test_predictions = model.predict(self.X_test)
        self.assertTrue(accuracy_score(self.Y_train, train_predictions) > 0)
        self.assertTrue(accuracy_score(self.Y_test, test_predictions) > 0)

    def test_k_nearest_neighbors(self):
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.Y_train)
        train_predictions = model.predict(self.X_train)
        test_predictions = model.predict(self.X_test)
        self.assertTrue(accuracy_score(self.Y_train, train_predictions) > 0)
        self.assertTrue(accuracy_score(self.Y_test, test_predictions) > 0)

    def test_decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.Y_train)
        train_predictions = model.predict(self.X_train)
        test_predictions = model.predict(self.X_test)
        self.assertTrue(accuracy_score(self.Y_train, train_predictions) > 0)
        self.assertTrue(accuracy_score(self.Y_test, test_predictions) > 0)

if __name__ == '__main__':
    unittest.main()
