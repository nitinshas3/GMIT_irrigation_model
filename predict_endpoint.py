from fastapi import FastAPI, Query
import requests
import numpy as np
import pickle
from datetime import datetime, timedelta

app = FastAPI(title="🌾 Smart Irrigation Prediction API", version="2.1 (WeatherAPI.com)")

# ✅ Load your trained model (ensure the file exists in the same directory)
model = pickle.load(open("xgb_irrigation_model.pkl", "rb"))

# --- WeatherAPI settings ---
WEATHER_API_KEY = "9980c416231943d3ba5132023250412"
FORECAST_URL = "https://api.weatherapi.com/v1/forecast.json"
HISTORY_URL = "https://api.weatherapi.com/v1/history.json"
HTTP_TIMEOUT = 15

latest_sensor_from_device = None

def _safe_get_totalprecip_mm(day_obj) -> float:
    """Defensive read of totalprecip_mm from a WeatherAPI 'day' object."""
    try:
        return float(day_obj.get("totalprecip_mm", 0.0))
    except Exception:
        return 0.0


def _history_precip_mm(lat: float, lon: float, date_str: str) -> float:
    """Get daily total precip (mm) for a past date using WeatherAPI history."""
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "dt": date_str,  # YYYY-MM-DD
        "aqi": "no",
        "alerts": "no",
    }
    r = requests.get(HISTORY_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # history returns one 'forecastday' with the requested date
    day = data["forecast"]["forecastday"][0]["day"]
    return _safe_get_totalprecip_mm(day)


# --- Helper: Fetch rainfall + compute bounded soil moisture change % ---
def fetch_weather_and_moisture_change(lat: float, lon: float):
    """
    Uses WeatherAPI:
      • forecast.json (today + next 3 days) for rainfall
      • history.json (previous 3 days) for past rainfall average
    Returns:
      rainfall_mm_today, rainfall_forecast_next_3days_mm, soil_moisture_change_percent
    """
    # ---- Forecast (today + next 3 days)
    f_params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "days": 4,   # today + next 3
        "aqi": "no",
        "alerts": "no",
    }
    f_res = requests.get(FORECAST_URL, params=f_params, timeout=HTTP_TIMEOUT)
    f_res.raise_for_status()
    f_json = f_res.json()

    forecast_days = f_json["forecast"]["forecastday"]
    # Today rainfall
    rainfall_mm_today = _safe_get_totalprecip_mm(forecast_days[0]["day"])

    # Next 3 days rainfall sum
    next3 = forecast_days[1:4] if len(forecast_days) > 1 else []
    rainfall_forecast_next_3days_mm = float(
        sum(_safe_get_totalprecip_mm(d["day"]) for d in next3)
    )

    # ---- Past 3 days average (history)
    today = datetime.utcnow().date()
    past_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in (1, 2, 3)]
    past_vals = []
    for ds in past_dates:
        try:
            past_vals.append(_history_precip_mm(lat, lon, ds))
        except Exception:
            # If any history call fails, skip it; we’ll fall back gracefully
            pass

    if len(past_vals) >= 1:
        past3_avg = float(np.mean(past_vals))
    else:
        # Fallback if history is unavailable: use today's as baseline
        past3_avg = float(rainfall_mm_today)

    # --- BOUNDED moisture-change proxy in [-20, +20]
    # Prevent blow-ups on low rain and keep output nicely bounded
    delta = rainfall_mm_today - past3_avg                    # mm
    scale = max(5.0, 0.5 * (rainfall_mm_today + past3_avg))  # mm; tune 5.0 / 0.5 for sensitivity
    soil_moisture_change_percent = 20.0 * float(np.tanh(delta / scale))

    return {
        "rainfall_mm_today": float(rainfall_mm_today),
        "rainfall_forecast_next_3days_mm": float(rainfall_forecast_next_3days_mm),
        "soil_moisture_change_percent": float(soil_moisture_change_percent),
    }


# --- Default root endpoint ---
@app.get("/")
def home():
    return {
        "message": "🌾 Welcome to the Smart Irrigation Prediction API (WeatherAPI.com)!",
        "endpoints": {
            "/weather": "Get live rainfall & forecast for given lat/long",
            "/predict": "Predict irrigation requirement based on live weather + soil data",
        },
    }


# --- Weather-only endpoint (for testing WeatherAPI fetch) ---
@app.get("/weather")
def get_weather(
    latitude: float = Query(14.45, description="Latitude (default Davangere)"),
    longitude: float = Query(75.90, description="Longitude (default Davangere)"),
):
    weather = fetch_weather_and_moisture_change(latitude, longitude)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "rainfall_mm_today": round(weather["rainfall_mm_today"], 2),
        "rainfall_forecast_next_3days_mm": round(weather["rainfall_forecast_next_3days_mm"], 2),
        "soil_moisture_change_percent": round(weather["soil_moisture_change_percent"], 2),
    }


# --- Main Prediction Endpoint ---
@app.post("/predict")
def predict_irrigation(
    latitude: float = Query(14.45, description="Latitude (default Davangere)"),
    longitude: float = Query(75.90, description="Longitude (default Davangere)"),
    soil_moisture_percent: float = Query(45.0, description="Soil moisture (%) from sensor"),
    soil_temperature_c: float = Query(28.0, description="Soil temperature (°C) from sensor"),
):
    """Fetch weather automatically, compute soil moisture change, and predict irrigation."""
    # 1) Weather + rainfall-derived proxy
    weather = fetch_weather_and_moisture_change(latitude, longitude)

    # 2) Features for model
    features = np.array([[
        latitude,
        longitude,
        soil_moisture_percent,
        soil_temperature_c,
        weather["soil_moisture_change_percent"],
        weather["rainfall_mm_today"],
        weather["rainfall_forecast_next_3days_mm"],
    ]])

    # 3) Model prediction
    prediction = float(model.predict(features)[0])
    prediction = max(0.0, prediction)  # clamp negatives

    # 4) Optional logging
    with open("prediction_logs.csv", "a") as f:
        f.write(
            f"{datetime.now()},{latitude},{longitude},{soil_moisture_percent},"
            f"{soil_temperature_c},{weather['rainfall_mm_today']},"
            f"{weather['rainfall_forecast_next_3days_mm']},{prediction}\n"
        )

    # 5) Response
    return {
        "latitude": latitude,
        "longitude": longitude,
        "rainfall_mm_today": round(weather["rainfall_mm_today"], 2),
        "rainfall_forecast_next_3days_mm": round(weather["rainfall_forecast_next_3days_mm"], 2),
        "soil_moisture_change_percent": round(weather["soil_moisture_change_percent"], 2),
        "predicted_irrigation_mm_day": round(prediction, 2),
        "message": "Prediction successful ✅",
    }

# Run with:
# uvicorn predict_endpoint:app --reload

latest_sensor_from_device = None

@app.post("/sensor")
def receive_sensor_data(sensor: dict):
    """
    Receives sensor data from the IoT device.
    Expected JSON:
    {
        "temperature": <float>,
        "humidity": <float>
    }
    """
    global latest_sensor_from_device

    if "temperature" not in sensor or "humidity" not in sensor:
        return {"error": "Missing temperature or humidity"}

    latest_sensor_from_device = {
        "temperature": float(sensor["temperature"]),
        "humidity": float(sensor["humidity"]),
        "soil_moisture_raw":float(sensor["soil_moisture_raw"]),
        "soil_moisture_percent": float(sensor["soil_moisture_percent"]),
        "timestamp": datetime.now().isoformat()
    }

    return {"status": "ok", "received": latest_sensor_from_device}