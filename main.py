import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Ibadan Property Price Predictor API",
    description="An API to predict property prices in Ibadan based on key features.",
    version="1.0.0"
)

# Load the trained model pipeline
try:
    model = joblib.load('models/best_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Please ensure 'best_model.pkl' is in the 'models/' directory.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# Define the input data model using Pydantic
class PropertyFeatures(BaseModel):
    Property_Type: str = Field(..., example="Duplex", description="Type of the property (e.g., Duplex, Bungalow, Flat).")
    Bedrooms: int = Field(..., example=4, description="Number of bedrooms.")
    Bathrooms: int = Field(..., example=5, description="Number of bathrooms.")
    Location: str = Field(..., example="Akobo", description="The location of the property within Ibadan.")

# Define the prediction response model
class PredictionResponse(BaseModel):
    predicted_price_ngn: float = Field(..., example=50000000.0, description="The predicted price of the property in Nigerian Naira (NGN).")

# Mount the 'static' directory to serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint to serve the HTML file
@app.get("/", include_in_schema=False)
async def read_index():
    """Serves the frontend HTML file."""
    return FileResponse('static/index.html')

# Endpoint to provide data for frontend forms
@app.get("/form-data", tags=["Frontend Support"])
def get_form_data():
    """Provides unique values for categorical features to populate dropdowns in the UI."""
    df = pd.read_csv('cleaned_ibadan_properties.csv')
    property_types = sorted(df['Property_Type'].unique().tolist())
    locations = sorted(df['Location'].unique().tolist())
    return {
        "property_types": property_types,
        "locations": locations
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_price(features: PropertyFeatures):
    """
    Predicts the price of a property based on its features.
    """
    if model is None:
        return {"error": "Model is not loaded. Cannot make predictions."}

    # Convert input features to a pandas DataFrame
    input_df = pd.DataFrame([features.dict()])

    # The model expects columns in a specific order, which the preprocessor handles.
    # The ColumnTransformer knows which columns are numeric and which are categorical.

    # Make a prediction (model predicts the log-transformed price)
    log_prediction = model.predict(input_df)

    # Inverse transform the prediction to get the actual price
    predicted_price = np.expm1(log_prediction)[0]

    return {"predicted_price_ngn": round(predicted_price, 2)}

# To run the app, use the command: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
