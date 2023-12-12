import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def preprocess_data(sample_customer_data, categorical_cols, label_encoder):
    # Extract relevant columns for prediction
    columns_for_prediction = ['Age', 'Gender', 'Category1', 'Category2', 'TotalItems', 'DiscountPercentage', 'TotalSpent']

    # Verify that the required columns are present
    missing_columns = set(columns_for_prediction) - set(sample_customer_data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in sample_customer_data.")

    sample_customer_data_relevant = sample_customer_data[columns_for_prediction].copy()

    # Label encode the 'ProductID' column
    sample_customer_data_relevant['ProductID'] = label_encoder.transform(sample_customer_data['ProductID'])

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

    return transformed_df


def get_top_recommendations(sample_customer_data, model_info, num_recommendations=10):
    # Assume you have a trained model
    pipeline = model_info['pipeline']
    label_encoder = model_info['label_encoder']
    categorical_cols = model_info['categorical_cols']

    # Preprocess the sample customer data
    transformed_data = preprocess_data(sample_customer_data, categorical_cols, label_encoder)

    # Define product IDs for which you want recommendations
    product_ids = ['A', 'B', 'C']

    # Repeat the sample customer data for each product
    repeated_data = np.repeat(transformed_data, len(product_ids), axis=0)

    # Repeat the product IDs for each row
    tiled_product_ids = np.tile(product_ids, len(sample_customer_data))

    # Extract relevant columns for prediction
    relevant_columns = ['CustomerID', 'TotalSpent', 'DiscountPercentage', 'TotalItems', 'Age']
    sample_customer_data_relevant = sample_customer_data[relevant_columns].copy()

    # Repeat the relevant columns for each product
    repeated_relevant_data = pd.concat([sample_customer_data_relevant] * len(product_ids), ignore_index=True)

    # Create a DataFrame with the expected column order for combined_data
    combined_data = pd.DataFrame()
    combined_data = pd.concat([combined_data, pd.DataFrame(repeated_data)], axis=1)
    combined_data = pd.concat([combined_data, repeated_relevant_data], axis=1)
    combined_data['ProductID'] = tiled_product_ids

    # Ensure 'ProductID' is treated as categorical during prediction
    combined_data['ProductID'] = combined_data['ProductID'].astype('category')

    # Predict purchase amounts
    predicted_purchase_amounts = pipeline.predict(combined_data)

    # Create DataFrame with results
    recommendation_data = pd.DataFrame({
        'CustomerID': np.repeat(sample_customer_data['CustomerID'].values, len(product_ids)),
        'ProductID': np.tile(product_ids, len(sample_customer_data)),
        'PredictedPurchaseAmount': predicted_purchase_amounts
    })

    # Sort products by predicted purchase amount in descending order
    recommendation_data = recommendation_data.sort_values(by='PredictedPurchaseAmount', ascending=False)

    # Display the top recommendations
    top_recommendations = recommendation_data.head(num_recommendations)
    print("Top Recommendations:")
    print(top_recommendations)

if __name__ == '__main__':
    # Load sample customer data
    sample_customer_data = pd.read_excel('sample_data.xlsx')

    # Load the trained model
    model_info = joblib.load('recommendation_model.pkl')

    # Get top recommendations for the sample customer
    get_top_recommendations(sample_customer_data, model_info, num_recommendations=10)
