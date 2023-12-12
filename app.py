# app.py

from flask import Flask, render_template
from recommendation_model import train_recommendation_model, get_recommendation_data
import sample_data_generator

app = Flask(__name__)

def generate_sample_data():
    """Step 1: Generate Sample Data."""
    print("Generating Sample Data...")
    sample_data_generator.generate_sample_data()

def collect_and_preprocess():
    """Step 2: Collect and Preprocess Data."""
    print("Collecting and Preprocessing Data...")
    structured_data = collect_and_preprocess_data()
    return structured_data

def train_model(structured_data):
    """Step 3: Train Recommendation Model."""
    print("Training Recommendation Model...")
    train_recommendation_model(structured_data)
    print("Model trained and saved successfully.")

def start_flask_app():
    """Step 4: Start Flask Web App."""
    print("Starting Flask Web App...")
    app.run(debug=True)

@app.route('/')
def index():
    """Fetch recommendation data and pass sample customer data to the template."""
    sample_customer_data = collect_and_preprocess()  # Assumes this provides sample data
    recommendation_data = get_recommendation_data(sample_customer_data)
    return render_template('index.html', recommendation_data=recommendation_data)

if __name__ == '__main__':
    generate_sample_data()
    structured_data = collect_and_preprocess()
    train_model(structured_data)
    start_flask_app()
