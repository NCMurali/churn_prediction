from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

class InputData(BaseModel):
    file: UploadFile

model = joblib.load("path_to_your_model.pkl")

# Preprocess function
def preprocess_data(df):
    # Remove any leading or trailing whitespace characters from column names
    df.columns = df.columns.str.strip()

    # Convert TotalCharges column to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values in TotalCharges column with the mean
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # Encode categorical columns using LabelEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Standardize numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

# Prediction function
def predict_churn(data):
    customer_ids = data['customerID'].copy()
    pred_data=preprocess_data(data)
    preds=model.predict(pred_data)
    data['PChurn']=preds
    predictions_df = pd.DataFrame({'customerID': customer_ids, 'PChurn': preds})
    sorted_predictions_df = predictions_df.sort_values(by='PChurn', ascending=False)
    top_10_rows = sorted_predictions_df.head(10) 
    
    return top_10_rows

# API endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file into a Pandas DataFrame
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(BytesIO(await file.read()))
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(await file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an Excel (XLSX) or CSV file.")

        # Call the prediction function
        result = predict_churn(df)

        # Convert result to dictionary format
        result_dict = result.to_dict(orient="records")

        return {"result": result_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
