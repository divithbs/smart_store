# recommendation_model.py

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# ...

def train_recommendation_model(structured_data):
    # Separate features and target variable
    # Separate features and target variable
    X = structured_data.drop('PredictedPurchaseAmount', axis=1)
    y = structured_data['PredictedPurchaseAmount']

    # Identify categorical columns for one-hot encoding
    categorical_cols = ['Gender', 'Category1', 'Category2']

    # Label encode the 'ProductID' column
    label_encoder = LabelEncoder()
    X['ProductID'] = label_encoder.fit_transform(X['ProductID'])

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols + ['ProductID'])
        ],
        remainder='passthrough'
    )

    # Create a pipeline with preprocessing and model
    model = RandomForestRegressor()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Fit the model with preprocessed data
    pipeline.fit(X, y)

    # Save the model, label encoder, and preprocessing information to a PKL file
    joblib.dump({'pipeline': pipeline, 'label_encoder': label_encoder, 'categorical_cols': categorical_cols}, 'recommendation_model.pkl')
    print("Model trained and saved successfully.")


def get_recommendation_data(sample_customer_data):
    # Assume you have a trained model
    model_info = joblib.load('recommendation_model.pkl')

    # Define product IDs for which you want recommendations
    product_ids = ['A', 'B', 'C']

    # Extract preprocessing information
    categorical_cols = model_info['categorical_cols']

    # Extract relevant columns for prediction
    # Extract relevant columns for prediction
    columns_for_prediction = ['Age', 'Gender', 'Category1', 'Category2', 'TotalItems', 'DiscountPercentage',
                              'TotalSpent']

    sample_customer_data_relevant = sample_customer_data[columns_for_prediction].copy()

    # Debug prints to identify the issue
    print("Columns in DataFrame:", sample_customer_data_relevant.columns)
    print("Sample Customer Data Relevant:")
    print(sample_customer_data_relevant.info())
    print(sample_customer_data_relevant.head())

    # Identify categorical columns for one-hot encoding
    categorical_cols = ['Gender', 'Category1', 'Category2']

    # Label encode the 'ProductID' column
    label_encoder = LabelEncoder()
    sample_customer_data_relevant['ProductID'] = label_encoder.fit_transform(sample_customer_data['ProductID'])

    # Encode all categorical columns using OneHotEncoder and scale numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(dtype='float64', handle_unknown='ignore'), categorical_cols + ['ProductID']),
            ('num', StandardScaler(), ~sample_customer_data_relevant.columns.isin(categorical_cols))
        ],
        remainder='passthrough'
    )

    # Apply fit_transform to all columns
    transformed_data = preprocessor.fit_transform(sample_customer_data_relevant)

    # Get the column names after encoding
    encoded_columns = preprocessor.get_feature_names_out(
        input_features=columns_for_prediction + ['ProductID'])

    # Create a DataFrame with the correct column names
    transformed_df = pd.DataFrame(transformed_data, columns=encoded_columns)

    # Repeat the sample customer data for each product
    repeated_data = np.repeat(transformed_df, len(product_ids), axis=0)

    # Repeat the product IDs for each row
    tiled_product_ids = np.tile(product_ids, len(sample_customer_data_relevant))

    # Combine the repeated data and tiled product IDs
    combined_data = pd.DataFrame(np.column_stack((repeated_data, tiled_product_ids)),
                                 columns=encoded_columns.tolist() + ['ProductID'])

    # Ensure 'ProductID' is treated as categorical during prediction
    combined_data['ProductID'] = combined_data['ProductID'].astype('category')

    # Ensure columns match the expected order and are present
    combined_data = combined_data[columns_for_prediction + ['ProductID']]

    # Debug prints to check combined_data
    print("Columns in Combined Data:", combined_data.columns)
    print("Combined Data:")
    print(combined_data.info())
    print(combined_data.head())

    # Predict purchase amounts
    predicted_purchase_amounts = model_info['pipeline'].predict(combined_data)

    # Create DataFrame with results
    recommendation_data = pd.DataFrame({
        'CustomerID': np.repeat(sample_customer_data['CustomerID'].values, len(product_ids)),
        'ProductID': np.tile(product_ids, len(sample_customer_data)),
        'PredictedPurchaseAmount': predicted_purchase_amounts
    })

    # Sort products by predicted purchase amount in descending order
    recommendation_data = recommendation_data.sort_values(by='PredictedPurchaseAmount', ascending=False)

    print(recommendation_data)

    return recommendation_data




def save_recommendation_result(recommendation_data):
    # Sort the DataFrame by 'PredictedPurchaseAmount' in descending order
    recommendation_data = recommendation_data.sort_values(by='PredictedPurchaseAmount', ascending=False)

    # Add a print statement to check the content of recommendation_data
    print("Recommendation Data:")
    print(recommendation_data)

    # Save the recommendation result to an Excel file
    recommendation_data.to_excel('recommendation_result.xlsx', index=False)
    print("Recommendation result saved to 'recommendation_result.xlsx'.")

# ...


if __name__ == '__main__':
    # Load sample customer data
    sample_customer_data = pd.read_excel('sample_data.xlsx')

    # Train the recommendation model
    train_recommendation_model(sample_customer_data)

    # Get recommendation data for the sample customer
    recommendation_data = get_recommendation_data(sample_customer_data)

    # Save the recommendation result to an Excel file
    save_recommendation_result(recommendation_data)
