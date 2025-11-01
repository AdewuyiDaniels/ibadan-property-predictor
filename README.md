# Ibadan Property Price Predictor

This project is a full-stack web application that predicts property prices in Ibadan, Nigeria, using a machine learning model. It features a data processing pipeline, model training, a FastAPI backend, and a simple web-based user interface.

This application was developed for **ExpertListingLimited**.

## Features

- **End-to-End ML Pipeline**: From raw data to a trained, serialized model.
- **Web Interface**: A user-friendly UI to input property details and receive a price prediction.
- **FastAPI Backend**: A robust and modern API to serve the model and the frontend.
- **Data-driven Dropdowns**: The UI's form dropdowns are dynamically populated from the dataset.

## Project Structure

```
.
├── .gitignore
├── main.py                 # FastAPI application
├── models/
│   └── best_model.pkl      # Saved scikit-learn pipeline for the best model
├── results/
│   └── experiment_results.json # Performance metrics for all trained models
├── scripts/
│   ├── data_cleaner.py     # Cleans raw Jiji data
│   ├── eda.py              # Performs exploratory data analysis
│   └── train.py            # Model training and evaluation pipeline
├── static/
│   └── index.html          # Frontend HTML, CSS, and JavaScript
├── data/
│   ├── jiji_ibadan_properties.csv # Raw scraped data used for training
│   └── cleaned_ibadan_properties.csv # Cleaned data
└── requirements.txt        # Python dependencies
```

*Note: The datasets are referenced in the structure but may need to be placed in a `data/` directory.*

## Setup and Installation

Follow these steps to set up and run the application locally.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## How to Run the Application

### 1. Run the Backend Server

Start the FastAPI application using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will start, and the application will be accessible in your web browser.

### 2. Access the Web Interface

Open your web browser and navigate to:

[**http://localhost:8000**](http://localhost:8000)

You can now use the form to get property price predictions.

## Model Performance

The model was trained on a dataset of 970 properties scraped from Jiji. After evaluating several algorithms (including RandomForest, XGBoost, and Ridge), the **Gradient Boosting Regressor** was selected as the best-performing model.

- **Final Test R-squared Score**: **0.51**

This score indicates that the model can explain approximately 51% of the variance in property prices in the test set, which is a solid result given the nature of the data.
