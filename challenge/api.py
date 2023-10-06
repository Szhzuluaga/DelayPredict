import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
#import pandas as pd
import joblib
from challenge.model import DelayModel
from typing import List
from mangum import Mangum

loaded_model = joblib.load('Modelo_entrenado.pkl')

app = FastAPI()
handler = Mangum(app=app)

delay_model=DelayModel()

operadores_validos = [
    "American Airlines",
    "Air Canada",
    "Air France",
    "Aeromexico",
    "Aerolineas Argentinas",
    "Austral",
    "Avianca",
    "Alitalia",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Iberia",
    "K.L.M.",
    "Qantas Airways",
    "United Airlines",
    "Grupo LATAM",
    "Sky Airline",
    "Latin American Wings",
    "Plus Ultra Lineas Aereas",
    "JetSmart SPA",
    "Oceanair Linhas Aereas",
    "Lacsa"
]

class InputFeatures(BaseModel):
    OPERA:str
    TIPOVUELO:str
    MES:int

class FlightRequest(BaseModel):
    flights : list[InputFeatures]

@app.get("/")
async def index():
    return {"message" : "go to /docs or /predict"}

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(features: FlightRequest) -> dict:
    predictions = []

    for flight in features.flights:
        if flight.OPERA not in operadores_validos:
            raise HTTPException(status_code=400, detail="OPERADOR INVALIDO")
        if flight.TIPOVUELO not in ["N", "I"]:
            raise HTTPException(status_code=400, detail="TIPO DE VUELO INVALIDO")
        if flight.MES not in range(1, 13):
            raise HTTPException(status_code=400, detail="MES INVALIDO")

        data_dict = {
            "OPERA": [flight.OPERA],
            "TIPOVUELO": [flight.TIPOVUELO],
            "MES": [flight.MES]
        }

        input_data = pd.DataFrame(data_dict)
        preprocessed_data = delay_model.preprocess_for_serving(input_data)
        prediction = loaded_model.predict(preprocessed_data)
        predictions.append(prediction[0])

    return {
        "predictions": predictions
    }


