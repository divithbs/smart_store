# recommendation_model.py

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def train_recommendation_model(structured_data):
    # Separate features and target variable
    X = structured_data.drop('PurchaseAmount', axis=1)
    y = structured_data['PurchaseAmount']

    # Identify categorical columns for one-hot encoding
    categorical_cols = ['ProductID', 'Gender', 'Category1', 'Category2']

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Create a pipeline with preprocessing and model
    model = RandomForestRegressor()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Fit the model with preprocessed data
    pipeline.fit(X, y)

    # Save the model to a PKL file
    joblib.dump(pipeline, 'recommendation_model.pkl')

# ...

# ...
# ...

def get_recommendation_data(sample_customer_data):
    # Assume you have a trained model
    model = joblib.load('recommendation_model.pkl')

    # Extract relevant columns for prediction
    columns_for_prediction = ['CustomerID', 'ProductID', 'Age', 'Gender', 'Category1', 'Category2', 'TotalItems',
                              'DiscountPercentage', 'TotalSpent']
    sample_customer_data_relevant = sample_customer_data[columns_for_prediction]

    # Encode 'ProductID' using LabelEncoder
    label_encoder = LabelEncoder()
    sample_customer_data_relevant['ProductID'] = label_encoder.fit_transform(sample_customer_data_relevant['ProductID'])

    # Predict purchase amounts for all products for the sample customer
    product_ids = ['A', 'B', 'C']  # Products for which you want recommendations
    num_samples = len(sample_customer_data_relevant)

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), [1, 3, 4, 5]),  # Assuming these are the indices of categorical columns
            ('num', StandardScaler(), [0, 2, 6, 7, 8])  # Assuming these are the indices of numeric columns
        ],
        remainder='passthrough'
    )

    # Fit and transform the sample_customer_data_relevant
    transformed_data = preprocessor.fit_transform(sample_customer_data_relevant)

    # Repeat the sample customer data for each product
    repeated_data = np.repeat(transformed_data, len(product_ids), axis=0)

    # Repeat the product IDs for each row
    tiled_product_ids = np.tile(product_ids, num_samples)

    # Combine the repeated data and tiled product ids
    combined_data = np.column_stack((repeated_data, tiled_product_ids))

    # Predict purchase amounts
    predicted_purchase_amounts = model.named_steps['model'].predict(combined_data)

    # Create DataFrame with results
    recommendation_data = pd.DataFrame({
        'CustomerID': np.repeat(sample_customer_data_relevant['CustomerID'].values, len(product_ids)),
        'ProductID': np.tile(product_ids, num_samples),
        'PredictedPurchaseAmount': predicted_purchase_amounts
    })

    # Sort products by predicted purchase amount in descending order
    recommendation_data = recommendation_data.sort_values(by='PredictedPurchaseAmount', ascending=False)

    return recommendation_data

# ...



def save_recommendation_result(recommendation_data):
    recommendation_data.to_excel('recommendation_result.xlsx', index=False)


if __name__ == '__main__':
    # Load sample customer data
    sample_customer_data = pd.read_excel('sample_data.xlsx')

    # Train the recommendation model
    train_recommendation_model(sample_customer_data)

    # Get recommendation data for the sample customer
    recommendation_data = get_recommendation_data(sample_customer_data)

    # Save the recommendation result to an Excel file
    save_recommendation_result(recommendation_data)
