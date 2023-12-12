# sample_data_generator.py

import pandas as pd
import numpy as np

def generate_sample_data(num_samples=5):
    # Generate sample structured data for initial testing

    data = {
        'CustomerID': np.arange(1, num_samples + 1),
        'ProductID': np.random.choice(['A', 'B', 'C'], num_samples),
        'PurchaseAmount': np.random.randint(10, 100, num_samples),
        'Age': np.random.randint(18, 65, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Category1': np.random.choice(['X', 'Y', 'Z'], num_samples),
        'Category2': np.random.choice(['P', 'Q', 'R'], num_samples),
        'TotalItems': np.random.randint(1, 10, num_samples),
        'DiscountPercentage': np.random.uniform(0, 0.3, num_samples),
        'TotalSpent': np.random.uniform(50, 200, num_samples)
    }

    sample_data = pd.DataFrame(data)
    sample_data.to_excel('sample_data.xlsx', index=False)

if __name__ == '__main__':
    generate_sample_data(num_samples=500)
