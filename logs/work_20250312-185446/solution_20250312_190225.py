class TestCase:
    def __init__(self, name, test_function, expected_outcome):
        self.name = name
        self.test_function = test_function
        self.expected_outcome = expected_outcome

    def execute(self):
        result = self.test_function()
        return result == self.expected_outcome


class TestingFramework:
    def __init__(self):
        self.results = []

    def execute_tests(self, test_cases):
        for test_case in test_cases:
            try:
                outcome = test_case.execute()
                self.results.append((test_case.name, outcome))
            except Exception as e:
                self.results.append((test_case.name, f"Error: {str(e)}"))

    def report(self):
        for name, result in self.results:
            print(f"Test '{name}': {'Passed' if result is True else 'Failed' if result is False else result}")


# Example test functions
def sample_test():
    return 5 + 5


def another_test():
    return 2 * 3


# Example usage
if __name__ == "__main__":
    test_cases = [
        TestCase("Sample Test", sample_test, 10),
        TestCase("Another Test", another_test, 6),
    ]

    framework = TestingFramework()
    framework.execute_tests(test_cases)
    framework.report()