import unittest
from predict import app


class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_predict_endpoint(self):
        response = self.app.get('/predict/10')
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', data)

    def test_predict_with_invalid_input(self):
        response = self.app.get('/predict/invalid_input')
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
