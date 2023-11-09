# objective_function.py
import numpy as np

def schwefel_function(x, y):
    return 418.9829 * 2 - x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))

# Add other functions as needed
