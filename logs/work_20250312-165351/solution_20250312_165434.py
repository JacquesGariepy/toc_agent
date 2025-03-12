import unittest

class ExampleTest(unittest.TestCase):
    
    def setUp(self):
        # Initialize necessary variables or configurations
        self.test_cases = self.create_test_cases()
    
    def create_test_cases(self):
        # Return a list of (input, expected_output) tuples
        return [
            (1, 2),   # Example input/output pair
            (2, 3),   # Example input/output pair
            (3, 4)    # Example input/output pair
        ]
    
    def example_function(self, x):
        # Define the function to be tested
        return x + 1
    
    def test_example_function(self):
        for input_value, expected_output in self.test_cases:
            with self.subTest(input=input_value):
                self.assertEqual(self.example_function(input_value), expected_output)

def main():
    unittest.main()

if __name__ == '__main__':
    main()