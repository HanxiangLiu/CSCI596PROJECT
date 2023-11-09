# objective_function.py
import numpy as np

def schwefel_function(x, y):
    return 418.9829 * 2 - x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))

# Add other functions as needed

if __name__ == "__main__":
    # Test the schwefel_function with known values
    x_test, y_test = 420.9687, 420.9687
    expected_result = 0
    result = schwefel_function(x_test, y_test)
    assert np.isclose(result, expected_result), f"Test failed! Expected {expected_result}, got {result}"
    print("objective_function.py tests passed.")