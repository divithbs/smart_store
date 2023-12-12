# app.py

from flask import Flask, render_template
from data_administrator import collect_and_preprocess_data
from recommendation_model import train_recommendation_model, get_recommendation_data
import sample_data_generator
import joblib

app = Flask(__name__)

def run_all_steps():
    # Step 1: Generate Sample Data
    print("Step 1: Generating Sample Data...")
    sample_data_generator.generate_sample_data()

    # Step 2: Collect and Preprocess Data
    print("Step 2: Collecting and Preprocessing Data...")
    structured_data = collect_and_preprocess_data()

    # Step 3: Train Recommendation Model
    print("Step 3: Training Recommendation Model...")
    train_recommendation_model(structured_data)
    print("Model trained and saved successfully.")

    # Step 4: Start Flask Web App
    print("Step 4: Starting Flask Web App...")
    app.run(debug=True)

@app.route('/')
def index():
    # Fetch recommendation data and pass it to the template
    recommendation_data = get_recommendation_data()
    return render_template('index.html', recommendation_data=recommendation_data)

if __name__ == '__main__':
    run_all_steps()
