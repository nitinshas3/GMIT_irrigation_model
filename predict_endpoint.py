from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta

app = FastAPI(title="🌾 Smart Irrigation Prediction API", version="2.2 (WeatherAPI.com + Sensor Integration)")

# --- CORS middleware for frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Bhuvan Micronutrient WMS Settings ---
WMS_BASE_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"

LAYER_MAP = {
    "ndvi": "bhuvan:LULC_250K",
    "vegetation": "lulc:LULC50K_1516",
    "lulc": "lulc:LULC50K_1516",
    "soil": "soil:SOIL_TEXTURE",
    "zinc": "bhuvan:INDIA_STATE",
    "iron": "bhuvan:INDIA_STATE",
    "boundary": "bhuvan:INDIA_STATE",
}


class BboxRequest(BaseModel):
    minLat: float
    minLon: float
    maxLat: float
    maxLon: float
    nutrient: str


# ✅ Load trained model (ensure the file exists in the same directory)
model = pickle.load(open("xgb_irrigation_model.pkl", "rb"))

# --- WeatherAPI settings ---
WEATHER_API_KEY = "9980c416231943d3ba5132023250412"
FORECAST_URL = "https://api.weatherapi.com/v1/forecast.json"
HISTORY_URL = "https://api.weatherapi.com/v1/history.json"
HTTP_TIMEOUT = 15

latest_sensor_from_device = None


# --- Utility functions ---
def _safe_get_totalprecip_mm(day_obj) -> float:
    try:
        return float(day_obj.get("totalprecip_mm", 0.0))
    except Exception:
        return 0.0


def _history_precip_mm(lat: float, lon: float, date_str: str) -> float:
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "dt": date_str,
        "aqi": "no",
        "alerts": "no",
    }
    r = requests.get(HISTORY_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    day = data["forecast"]["forecastday"][0]["day"]
    return _safe_get_totalprecip_mm(day)


# --- Helper: Fetch rainfall and soil moisture change ---
def fetch_weather_and_moisture_change(lat: float, lon: float):
    f_params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "days": 4,
        "aqi": "no",
        "alerts": "no",
    }
    f_res = requests.get(FORECAST_URL, params=f_params, timeout=HTTP_TIMEOUT)
    f_res.raise_for_status()
    f_json = f_res.json()

    forecast_days = f_json["forecast"]["forecastday"]
    rainfall_mm_today = _safe_get_totalprecip_mm(forecast_days[0]["day"])
    next3 = forecast_days[1:4] if len(forecast_days) > 1 else []
    rainfall_forecast_next_3days_mm = float(sum(_safe_get_totalprecip_mm(d["day"]) for d in next3))

    today = datetime.utcnow().date()
    past_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in (1, 2, 3)]
    past_vals = []
    for ds in past_dates:
        try:
            past_vals.append(_history_precip_mm(lat, lon, ds))
        except Exception:
            pass

    past3_avg = float(np.mean(past_vals)) if past_vals else rainfall_mm_today
    delta = rainfall_mm_today - past3_avg
    scale = max(5.0, 0.5 * (rainfall_mm_today + past3_avg))
    soil_moisture_change_percent = 20.0 * float(np.tanh(delta / scale))

    return {
        "rainfall_mm_today": float(rainfall_mm_today),
        "rainfall_forecast_next_3days_mm": float(rainfall_forecast_next_3days_mm),
        "soil_moisture_change_percent": float(soil_moisture_change_percent),
    }


# --- Root Endpoint ---
@app.get("/")
def home():
    return {
        "message": "🌾 Welcome to the Smart Irrigation Prediction API (WeatherAPI.com)!",
        "endpoints": {
            "/weather": "Get live rainfall & forecast for given lat/long",
            "/predict": "Predict irrigation requirement based on live weather + soil data",
            "/sensor (POST)": "Receive and save sensor data",
            "/sensor (GET)": "Retrieve latest sensor data",
            "/api/micronutrient-map (POST)": "Get zinc/iron/NDVI heatmap for bounding box",
        },
    }


# --- Weather-only Endpoint ---
@app.get("/weather")
def get_weather(latitude: float = Query(14.45), longitude: float = Query(75.90)):
    weather = fetch_weather_and_moisture_change(latitude, longitude)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "rainfall_mm_today": round(weather["rainfall_mm_today"], 2),
        "rainfall_forecast_next_3days_mm": round(weather["rainfall_forecast_next_3days_mm"], 2),
        "soil_moisture_change_percent": round(weather["soil_moisture_change_percent"], 2),
    }


# --- Receive Sensor Data ---
@app.post("/sensor")
def receive_sensor_data(sensor: dict):
    global latest_sensor_from_device
    required_keys = ["temperature", "humidity", "soil_moisture_raw", "soil_moisture_percent"]

    if not all(k in sensor for k in required_keys):
        raise HTTPException(status_code=400, detail="Missing required sensor fields")

    latest_sensor_from_device = {
        "temperature": float(sensor["temperature"]),
        "humidity": float(sensor["humidity"]),
        "soil_moisture_raw": float(sensor["soil_moisture_raw"]),
        "soil_moisture_percent": float(sensor["soil_moisture_percent"]),
        "timestamp": datetime.now().isoformat()
    }

    # Save latest sensor data to JSON file
    with open("latest_sensor.json", "w") as f:
        json.dump(latest_sensor_from_device, f, indent=4)

    return {"status": "ok", "received": latest_sensor_from_device}


# --- Get Latest Sensor Data ---
@app.get("/sensor")
def get_latest_sensor_data():
    global latest_sensor_from_device
    if latest_sensor_from_device is None:
        if os.path.exists("latest_sensor.json"):
            with open("latest_sensor.json", "r") as f:
                latest_sensor_from_device = json.load(f)
        else:
            return {"error": "No sensor data available yet", "status": "no_data"}

    return {"status": "ok", "data": latest_sensor_from_device}


# --- Predict Endpoint ---
@app.post("/predict")
def predict_irrigation(
    latitude: float = Query(14.45),
    longitude: float = Query(75.90),
    soil_moisture_percent: float | None = Query(None),
    soil_temperature_c: float | None = Query(None),
):
    # Load latest sensor data if missing
    if (soil_moisture_percent is None or soil_temperature_c is None) and os.path.exists("latest_sensor.json"):
        with open("latest_sensor.json", "r") as f:
            sensor_data = json.load(f)
        soil_moisture_percent = soil_moisture_percent or sensor_data.get("soil_moisture_percent", 45.0)
        soil_temperature_c = soil_temperature_c or sensor_data.get("temperature", 28.0)
    else:
        soil_moisture_percent = soil_moisture_percent or 45.0
        soil_temperature_c = soil_temperature_c or 28.0

    weather = fetch_weather_and_moisture_change(latitude, longitude)

    features = np.array([[
        latitude,
        longitude,
        soil_moisture_percent,
        soil_temperature_c,
        weather["soil_moisture_change_percent"],
        weather["rainfall_mm_today"],
        weather["rainfall_forecast_next_3days_mm"],
    ]])

    prediction = float(model.predict(features)[0])
    prediction = max(0.0, prediction)

    with open("prediction_logs.csv", "a") as f:
        f.write(
            f"{datetime.now()},{latitude},{longitude},{soil_moisture_percent},"
            f"{soil_temperature_c},{weather['rainfall_mm_today']},"
            f"{weather['rainfall_forecast_next_3days_mm']},{prediction}\n"
        )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "soil_moisture_percent": soil_moisture_percent,
        "soil_temperature_c": soil_temperature_c,
        "rainfall_mm_today": round(weather["rainfall_mm_today"], 2),
        "rainfall_forecast_next_3days_mm": round(weather["rainfall_forecast_next_3days_mm"], 2),
        "soil_moisture_change_percent": round(weather["soil_moisture_change_percent"], 2),
        "predicted_irrigation_mm_day": round(prediction, 2),
        "message": "Prediction successful ✅ (auto sensor input used if available)",
    }


# --- Micronutrient Heatmap ---
@app.post("/api/micronutrient-map")
def get_micronutrient_map(req: BboxRequest):
    layer = LAYER_MAP.get(req.nutrient.lower(), "bhuvan:INDIA_STATE")
    bbox = f"{req.minLon},{req.minLat},{req.maxLon},{req.maxLat}"
    image_url = (
        f"{WMS_BASE_URL}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}&"
        f"BBOX={bbox}&WIDTH=512&HEIGHT=512&SRS=EPSG:4326&FORMAT=image/png&TRANSPARENT=true"
    )
    return {"imageUrl": image_url, "layer": layer, "nutrient": req.nutrient}


# --- WMS Capabilities ---
@app.get("/api/wms-capabilities")
def get_wms_capabilities():
    return {
        "capabilities_url": f"{WMS_BASE_URL}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities",
        "available_layers": list(LAYER_MAP.keys()),
        "layer_details": LAYER_MAP,
    }
