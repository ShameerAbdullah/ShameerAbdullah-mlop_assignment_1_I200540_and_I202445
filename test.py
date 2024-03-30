import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from task import evaluate_model


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        # Load the dataset
        self.titanic_data = pd.read_csv('titanic.csv')

        # Preprocessing
        self.titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
        self.titanic_data = self.titanic_data.drop(columns='Cabin', axis=1)
        self.titanic_data['Age'].fillna(self.titanic_data['Age'].mean(), inplace=True)
        self.titanic_data['Embarked'].fillna(self.titanic_data['Embarked'].mode()[0], inplace=True)

        # Split data into train and test sets
        self.X_processed = self.titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
        self.Y_processed = self.titanic_data['Survived']
        self.X_train_processed, self.X_test_processed, self.Y_train_processed, self.Y_test_processed = train_test_split(self.X_processed, self.Y_processed, test_size=0.2, random_state=2)

    def test_evaluate_model(self):
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train_processed, self.Y_train_processed)
        train_acc, test_acc = evaluate_model(model, self.X_train_processed, self.Y_train_processed, self.X_test_processed, self.Y_test_processed)
        self.assertIsInstance(train_acc, float)
        self.assertIsInstance(test_acc, float)
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertLessEqual(train_acc, 1.0)
        self.assertGreaterEqual(test_acc, 0.0)
        self.assertLessEqual(test_acc, 1.0)


if __name__ == '__main__':
    unittest.main()
