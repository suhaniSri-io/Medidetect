"""
AI MediDetect - Flask Application

Educational symptom-pattern classification and healthcare-assistance project.

Important:
This application is an educational decision-support prototype.
It is not a medical diagnosis or prescription service.
For severe symptoms, chest pain, breathing difficulty, unconsciousness,
or suspected emergency, seek urgent medical care immediately.
"""

from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import requests
from flask import Flask, flash, redirect, render_template, request, url_for


# -------------------------------------------------------------------
# Paths and application configuration
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH_IMPROVED = MODEL_DIR / "disease_model_improved.pkl"
ENCODER_PATH_IMPROVED = MODEL_DIR / "feature_encoder_improved.pkl"

MODEL_PATH_BACKUP = MODEL_DIR / "disease_model.pkl"
ENCODER_PATH_BACKUP = MODEL_DIR / "feature_encoder.pkl"

DISEASES_PATH = DATA_DIR / "diseases.csv"
HISTORY_PATH = DATA_DIR / "history.csv"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",           # Primary
    "https://overpass.kumi.systems/api/interpreter",     # Fallback 1
    "https://z.overpass-api.de/api/interpreter",         # Fallback 2
]

DEFAULT_RADIUS = 5000
MIN_RADIUS = 500
MAX_RADIUS = 50000

USER_AGENT = (
    "AI-MediDetect/2.0 "
    "(educational healthcare project; contact project-maintainer)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("ai_medidetect")

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-secret-key-before-deployment",
)


# -------------------------------------------------------------------
# Symptom display labels
# -------------------------------------------------------------------

SYMPTOMS_LIST = [
    "fever",
    "headache",
    "cough",
    "fatigue",
    "nausea",
    "vomiting",
    "sore_throat",
    "body_pain",
    "chest_pain",
    "shortness_of_breath",
    "dizziness",
    "joint_pain",
    "runny_nose",
    "chills",
    "loss_of_appetite",
    "muscle_pain",
    "weakness",
    "sneezing",
    "wheezing",
]

SYMPTOMS_DISPLAY = {
    "fever": "Fever",
    "headache": "Headache",
    "cough": "Cough",
    "fatigue": "Fatigue",
    "nausea": "Nausea",
    "vomiting": "Vomiting",
    "sore_throat": "Sore Throat",
    "body_pain": "Body Pain",
    "chest_pain": "Chest Pain",
    "shortness_of_breath": "Shortness of Breath",
    "dizziness": "Dizziness",
    "joint_pain": "Joint Pain",
    "runny_nose": "Runny Nose",
    "chills": "Chills",
    "loss_of_appetite": "Loss of Appetite",
    "muscle_pain": "Muscle Pain",
    "weakness": "Weakness",
    "sneezing": "Sneezing",
    "wheezing": "Wheezing",
}


# -------------------------------------------------------------------
# Default hospital recommendations
# -------------------------------------------------------------------

DEFAULT_DISEASE_HOSPITAL_MAP = {
    "common cold": {
        "priority": "general",
        "specialties": ["general practice", "clinic", "urgent care"],
        "urgency": "low",
        "advice": "Visit a general clinic or see a doctor if symptoms persist or worsen.",
    },
    "viral fever": {
        "priority": "general",
        "specialties": ["general hospital", "fever clinic", "clinic"],
        "urgency": "low",
        "advice": "Consider a clinic or general hospital assessment, especially if fever persists.",
    },
    "pneumonia": {
        "priority": "high",
        "specialties": ["chest hospital", "pulmonology", "general hospital"],
        "urgency": "high",
        "advice": "Seek prompt medical assessment, particularly for breathing difficulty or chest pain.",
    },
    "asthma": {
        "priority": "medium",
        "specialties": ["respiratory clinic", "pulmonology", "general hospital"],
        "urgency": "medium",
        "advice": "Consult a respiratory specialist or clinician for assessment.",
    },
    "heart disease": {
        "priority": "critical",
        "specialties": ["cardiology", "cardiac hospital", "emergency"],
        "urgency": "critical",
        "advice": "For chest pain or breathing difficulty, seek emergency medical care immediately.",
    },
    "diabetes": {
        "priority": "medium",
        "specialties": ["endocrinology", "general hospital", "clinic"],
        "urgency": "medium",
        "advice": "Consult a clinician for appropriate blood-sugar testing and management.",
    },
    "allergic rhinitis": {
        "priority": "low",
        "specialties": ["clinic", "general practice", "ENT"],
        "urgency": "low",
        "advice": "Visit a clinic or ENT specialist if symptoms are persistent.",
    },
}


# -------------------------------------------------------------------
# Model global variables
# -------------------------------------------------------------------

model = None
label_encoder = None
symptom_columns = []
diseases_hospital_map = DEFAULT_DISEASE_HOSPITAL_MAP.copy()
diseases_info = {}
loaded_model_path = None


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

def get_model_paths():
    """
    Select one matching model and encoder pair.

    The improved pair is used only when BOTH improved files exist.
    Otherwise, the backup pair is used only when BOTH backup files exist.
    """
    improved_pair_exists = (
        MODEL_PATH_IMPROVED.exists()
        and ENCODER_PATH_IMPROVED.exists()
    )

    backup_pair_exists = (
        MODEL_PATH_BACKUP.exists()
        and ENCODER_PATH_BACKUP.exists()
    )

    if improved_pair_exists:
        return MODEL_PATH_IMPROVED, ENCODER_PATH_IMPROVED

    if backup_pair_exists:
        return MODEL_PATH_BACKUP, ENCODER_PATH_BACKUP

    raise FileNotFoundError(
        "No complete model pair was found. Expected either:\n"
        f"1. {MODEL_PATH_IMPROVED.name} and "
        f"{ENCODER_PATH_IMPROVED.name}\n"
        "or\n"
        f"2. {MODEL_PATH_BACKUP.name} and "
        f"{ENCODER_PATH_BACKUP.name}"
    )


def load_model_files():
    """
    Load trained model, label encoder, feature columns, and hospital metadata.
    """
    global model
    global label_encoder
    global symptom_columns
    global diseases_hospital_map
    global loaded_model_path

    model_path, encoder_path = get_model_paths()

    logger.info("Loading model from: %s", model_path)
    logger.info("Loading encoder from: %s", encoder_path)

    with model_path.open("rb") as file:
        loaded_model = pickle.load(file)

    with encoder_path.open("rb") as file:
        encoder_data = pickle.load(file)

    if not isinstance(encoder_data, dict):
        raise TypeError(
            "The encoder file must contain a dictionary. "
            f"Found: {type(encoder_data).__name__}"
        )

    loaded_label_encoder = encoder_data.get("label_encoder")
    loaded_symptom_columns = encoder_data.get("symptom_columns")

    if loaded_label_encoder is None:
        raise ValueError(
            "The encoder file does not contain 'label_encoder'."
        )

    if not loaded_symptom_columns:
        raise ValueError(
            "The encoder file does not contain 'symptom_columns'."
        )

    loaded_symptom_columns = [
        str(symptom).strip().lower()
        for symptom in loaded_symptom_columns
    ]

    if len(set(loaded_symptom_columns)) != len(loaded_symptom_columns):
        raise ValueError(
            "Duplicate symptom columns were found in the encoder file."
        )

    model_feature_names = getattr(
        loaded_model,
        "feature_names_in_",
        None,
    )

    if model_feature_names is not None:
        model_feature_names = [
            str(feature).strip().lower()
            for feature in model_feature_names
        ]

        if model_feature_names != loaded_symptom_columns:
            raise ValueError(
                "Model feature order does not match the encoder symptom "
                "columns. Retrain and save both files together."
            )

    expected_feature_count = getattr(
        loaded_model,
        "n_features_in_",
        len(loaded_symptom_columns),
    )

    if expected_feature_count != len(loaded_symptom_columns):
        raise ValueError(
            "Model feature count does not match encoder symptom count. "
            f"Model expects {expected_feature_count}, but encoder has "
            f"{len(loaded_symptom_columns)}."
        )

    model = loaded_model
    label_encoder = loaded_label_encoder
    symptom_columns = loaded_symptom_columns
    diseases_hospital_map = encoder_data.get(
        "disease_hospital_map",
        DEFAULT_DISEASE_HOSPITAL_MAP,
    )
    loaded_model_path = str(model_path)

    logger.info("Model loaded successfully.")
    logger.info("Model type: %s", type(model).__name__)
    logger.info("Feature count: %d", len(symptom_columns))
    logger.info("Features: %s", symptom_columns)


try:
    load_model_files()
except Exception as exc:
    model = None
    label_encoder = None
    symptom_columns = []
    loaded_model_path = None

    logger.exception(
        "Model loading failed. Predictions will not be available: %s",
        exc,
    )


# -------------------------------------------------------------------
# Disease information and safe self-care information
# -------------------------------------------------------------------

def load_disease_information():
    global diseases_info

    if not DISEASES_PATH.exists():
        logger.warning(
            "Disease information CSV was not found: %s",
            DISEASES_PATH,
        )
        diseases_info = {}
        return

    try:
        diseases_df = pd.read_csv(DISEASES_PATH)

        if "disease" not in diseases_df.columns:
            raise ValueError(
                "diseases.csv must contain a column named 'disease'."
            )

        diseases_df["disease"] = (
            diseases_df["disease"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        diseases_info = diseases_df.set_index(
            "disease"
        ).to_dict("index")

        logger.info(
            "Loaded information for %d diseases.",
            len(diseases_info),
        )

    except Exception as exc:
        logger.exception(
            "Could not load disease information: %s",
            exc,
        )
        diseases_info = {}


load_disease_information()


def get_disease_info(disease_name):
    disease_key = str(disease_name).strip().lower()
    return diseases_info.get(disease_key, {})


MEDICINE_RULES = {
    "common cold": {
        "medicines": [
            "Rest, warm fluids, and saline nasal spray may provide supportive relief.",
            "Paracetamol may help fever or body pain when suitable for the person.",
        ],
        "precautions": [
            "Follow the medicine label and avoid duplicate paracetamol-containing products.",
            "Do not start antibiotics without advice from a qualified clinician.",
        ],
        "red_flags": [
            "Difficulty breathing",
            "Chest pain",
            "Confusion",
            "Symptoms that worsen or do not improve",
        ],
    },
    "viral fever": {
        "medicines": [
            "Paracetamol may help fever or body pain when suitable for the person.",
            "Drink fluids or oral rehydration solution to prevent dehydration.",
        ],
        "precautions": [
            "Rest and monitor your temperature.",
            "Do not exceed the recommended medicine dose.",
            "Do not use antibiotics unless prescribed by a clinician.",
        ],
        "red_flags": [
            "Very high fever",
            "Fever lasting more than three days",
            "Severe weakness, confusion, breathing difficulty, or dehydration",
        ],
    },
    "allergic rhinitis": {
        "medicines": [
            "Avoid known triggers such as dust, pollen, smoke, or allergens.",
            "Saline nasal spray may provide supportive relief.",
        ],
        "precautions": [
            "Some allergy medicines can cause drowsiness.",
            "Consult a pharmacist or clinician before using medicines if pregnant, breastfeeding, or taking other medicines.",
        ],
        "red_flags": [
            "Swelling of the face, lips, or tongue",
            "Wheezing or breathing difficulty",
            "Fainting or severe dizziness",
        ],
    },
    "asthma": {
        "medicines": [
            "Use only medicines or inhalers prescribed for you by a qualified clinician.",
            "Avoid smoke, dust, and other known triggers where possible.",
        ],
        "precautions": [
            "Do not share inhalers or use another person's prescription medicine.",
            "Seek medical evaluation for recurring wheeze or shortness of breath.",
        ],
        "red_flags": [
            "Severe or worsening shortness of breath",
            "Blue lips or face",
            "Difficulty speaking because of breathlessness",
        ],
    },
}


def normalize_text(value):
    return " ".join(str(value).strip().lower().split())


def get_health_advice(disease_name, disease_info=None):
    disease_key = normalize_text(disease_name)
    disease_info = disease_info or {}

    if disease_key in MEDICINE_RULES:
        return MEDICINE_RULES[disease_key]

    medicines = (
        disease_info.get("medicines")
        or disease_info.get("medicine")
        or disease_info.get("medication")
        or "Consult a qualified healthcare professional for treatment advice."
    )

    precautions = (
        disease_info.get("precautions")
        or disease_info.get("precaution")
        or "Rest, stay hydrated, and monitor your symptoms."
    )

    red_flags = (
        disease_info.get("red_flags")
        or disease_info.get("warning_signs")
        or "Seek medical care if symptoms become severe or worsen."
    )

    def convert_to_list(value):
        if isinstance(value, list):
            return value

        return [
            item.strip()
            for item in str(value).split("|")
            if item.strip()
        ]

    return {
        "medicines": convert_to_list(medicines),
        "precautions": convert_to_list(precautions),
        "red_flags": convert_to_list(red_flags),
    }


# -------------------------------------------------------------------
# Prediction and prediction history
# -------------------------------------------------------------------

def save_prediction(symptoms, predicted_disease, confidence):
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

        history_row = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "symptoms": ",".join(symptoms),
            "predicted_disease": predicted_disease,
            "confidence": f"{confidence:.1f}%",
        }

        if HISTORY_PATH.exists():
            history_df = pd.read_csv(HISTORY_PATH)
            history_df = pd.concat(
                [history_df, pd.DataFrame([history_row])],
                ignore_index=True,
            )
        else:
            history_df = pd.DataFrame([history_row])

        history_df.to_csv(HISTORY_PATH, index=False)
        return True

    except Exception as exc:
        logger.exception("Could not save prediction history: %s", exc)
        return False


def get_prediction_history():
    try:
        if not HISTORY_PATH.exists():
            return []

        return pd.read_csv(HISTORY_PATH).to_dict("records")

    except Exception as exc:
        logger.exception("Could not load history: %s", exc)
        return []


def make_feature_vector(selected_symptoms):
    """
    Build a one-row DataFrame with exactly the saved training feature names
    and order.
    """
    if not symptom_columns:
        raise RuntimeError(
            "Model symptom columns are unavailable."
        )

    selected_set = {
        str(symptom).strip().lower()
        for symptom in selected_symptoms
    }

    feature_row = {
        column: int(column in selected_set)
        for column in symptom_columns
    }

    return pd.DataFrame(
        [feature_row],
        columns=symptom_columns,
    )


def decode_disease_label(class_value):
    """
    Convert the model output into a disease name.

    The corrected training file uses LabelEncoder, so model outputs are
    integer class labels. The fallback supports models trained on strings.
    """
    if label_encoder is None:
        return str(class_value)

    if isinstance(class_value, (int, np.integer)):
        return str(
            label_encoder.inverse_transform(
                [int(class_value)]
            )[0]
        )

    return str(class_value)


def get_top_predictions(probabilities, limit=3):
    probabilities = np.asarray(probabilities, dtype=float)

    class_values = getattr(
        model,
        "classes_",
        np.arange(len(probabilities)),
    )

    ranked_positions = np.argsort(probabilities)[::-1][:limit]
    predictions = []

    for position in ranked_positions:
        class_value = class_values[position]

        predictions.append(
            {
                "disease": decode_disease_label(class_value),
                "probability": round(
                    float(probabilities[position] * 100),
                    2,
                ),
            }
        )

    return predictions


def get_hospital_recommendations(disease_name):
    disease_key = normalize_text(disease_name)

    for map_key, recommendation in diseases_hospital_map.items():
        normalized_map_key = normalize_text(map_key)

        if (
            normalized_map_key == disease_key
            or normalized_map_key in disease_key
            or disease_key in normalized_map_key
        ):
            return recommendation

    return {
        "priority": "medium",
        "specialties": ["general hospital", "clinic"],
        "urgency": "medium",
        "advice": (
            "Consult a qualified healthcare professional "
            "for an appropriate assessment."
        ),
    }


# -------------------------------------------------------------------
# Hospital search functions
# -------------------------------------------------------------------

def valid_coordinates(lat, lon):
    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError):
        return None

    if not -90 <= latitude <= 90:
        return None

    if not -180 <= longitude <= 180:
        return None

    return latitude, longitude


def clamp_radius(radius):
    try:
        radius = int(radius)
    except (TypeError, ValueError):
        radius = DEFAULT_RADIUS

    return max(MIN_RADIUS, min(radius, MAX_RADIUS))


def geocode_location(location_query):
    if not location_query or not location_query.strip():
        return {
            "error": "Please enter a city, area, address, or pincode."
        }

    try:
        response = requests.get(
            NOMINATIM_URL,
            params={
                "q": location_query.strip(),
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
            },
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "en",
            },
            timeout=20,
        )

        if response.status_code != 200:
            return {
                "error": (
                    "Location service returned an error. "
                    "Please try a more specific location."
                )
            }

        results = response.json()

        if not results:
            return {
                "error": (
                    f"Location '{location_query}' was not found. "
                    "Try a city, area, address, or pincode."
                )
            }

        result = results[0]
        coordinates = valid_coordinates(
            result.get("lat"),
            result.get("lon"),
        )

        if coordinates is None:
            return {
                "error": (
                    "The location service returned invalid coordinates."
                )
            }

        return {
            "lat": coordinates[0],
            "lon": coordinates[1],
            "display_name": result.get(
                "display_name",
                location_query,
            ),
        }

    except requests.exceptions.Timeout:
        return {
            "error": (
                "Location search timed out. Please try again."
            )
        }

    except requests.exceptions.RequestException:
        return {
            "error": (
                "Could not connect to the location service. "
                "Check your internet connection."
            )
        }


def build_overpass_query(lat, lon, radius):
    return f"""
[out:json];
(
  node["amenity"="hospital"](around:{radius},{lat},{lon});
  way["amenity"="hospital"](around:{radius},{lat},{lon});
  relation["amenity"="hospital"](around:{radius},{lat},{lon});

  node["amenity"="clinic"](around:{radius},{lat},{lon});
  way["amenity"="clinic"](around:{radius},{lat},{lon});
  relation["amenity"="clinic"](around:{radius},{lat},{lon});
);
out center;
"""


def calculate_distance_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    value = (
        sin(delta_lat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(delta_lon / 2) ** 2
    )

    return earth_radius_km * (
        2 * atan2(sqrt(value), sqrt(1 - value))
    )


def get_element_coordinates(element, fallback_lat, fallback_lon):
    if element.get("center"):
        return (
            element["center"].get("lat", fallback_lat),
            element["center"].get("lon", fallback_lon),
        )

    return (
        element.get("lat", fallback_lat),
        element.get("lon", fallback_lon),
    )


def format_address(tags, location_query=None):
    if tags.get("addr:full"):
        return tags["addr:full"]

    address_parts = []

    house_number = tags.get("addr:housenumber", "")
    street = tags.get("addr:street", "")

    if house_number and street:
        address_parts.append(f"{house_number} {street}")
    elif street:
        address_parts.append(street)

    for key in ("addr:suburb", "addr:city", "addr:state"):
        if tags.get(key):
            address_parts.append(tags[key])

    if tags.get("addr:postcode"):
        address_parts.append(tags["addr:postcode"])

    if address_parts:
        return ", ".join(address_parts)

    if location_query:
        return f"Near {location_query}"

    return "Address not available"


def request_overpass(query):
    failures = []

    for overpass_url in OVERPASS_URLS:
        try:
            response = requests.post(
                overpass_url,
                data={"data": query},
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
                timeout=(15, 60),
            )

            if response.status_code == 200:
                return response.json()

            failures.append(
                f"{overpass_url} returned HTTP "
                f"{response.status_code}"
            )

        except requests.exceptions.Timeout:
            failures.append(f"{overpass_url} timed out")

        except requests.exceptions.RequestException as exc:
            failures.append(
                f"{overpass_url} connection failed: {exc}"
            )

        time.sleep(1)

    raise RuntimeError(" | ".join(failures))


def find_nearby_hospitals(
    location_query=None,
    lat=None,
    lon=None,
    radius=DEFAULT_RADIUS,
):
    radius = clamp_radius(radius)

    coordinates = valid_coordinates(lat, lon)

    if coordinates is None:
        geocoded = geocode_location(location_query)

        if "error" in geocoded:
            return geocoded

        lat = geocoded["lat"]
        lon = geocoded["lon"]
        search_location = geocoded["display_name"]

    else:
        lat, lon = coordinates
        search_location = location_query or f"{lat}, {lon}"

    query = build_overpass_query(lat, lon, radius)

    try:
        data = request_overpass(query)
    except Exception as exc:
        logger.exception("Hospital search failed: %s", exc)
        return {
            "error": (
                "Hospital data service is temporarily unavailable. "
                "Please try again shortly."
            )
        }

    hospitals = []
    seen = set()

    for element in data.get("elements", []):
        tags = element.get("tags", {})

        hospital_coordinates = get_element_coordinates(
            element,
            lat,
            lon,
        )

        valid_hospital_coordinates = valid_coordinates(
            hospital_coordinates[0],
            hospital_coordinates[1],
        )

        if valid_hospital_coordinates is None:
            continue

        hospital_lat, hospital_lon = valid_hospital_coordinates

        name = (
            tags.get("name")
            or tags.get("official_name")
            or tags.get("operator")
            or "Unnamed healthcare facility"
        )

        unique_key = (
            str(name).strip().lower(),
            round(hospital_lat, 5),
            round(hospital_lon, 5),
        )

        if unique_key in seen:
            continue

        seen.add(unique_key)

        distance_km = calculate_distance_km(
            lat,
            lon,
            hospital_lat,
            hospital_lon,
        )

        hospitals.append(
            {
                "name": name,
                "lat": hospital_lat,
                "lon": hospital_lon,
                "address": format_address(
                    tags,
                    location_query,
                ),
                "phone": (
                    tags.get("phone")
                    or tags.get("contact:phone")
                    or ""
                ),
                "emergency": (
                    tags.get("emergency", "").lower() == "yes"
                ),
                "website": (
                    tags.get("website")
                    or tags.get("contact:website")
                    or ""
                ),
                "type": (
                    tags.get("amenity")
                    or tags.get("healthcare")
                    or "healthcare facility"
                ),
                "distance": f"{distance_km:.2f} km",
                "distance_value": distance_km,
            }
        )

    hospitals.sort(key=lambda item: item["distance_value"])

    for hospital in hospitals:
        hospital.pop("distance_value", None)

    if not hospitals:
        return {
            "error": (
                f"No healthcare facilities were found within "
                f"{radius / 1000:.1f} km of {search_location}. "
                "Try increasing the radius or using another nearby location."
            )
        }

    return {
        "hospitals": hospitals[:50],
        "center": {"lat": lat, "lon": lon},
        "count": len(hospitals),
        "search_location": search_location,
        "radius_km": radius / 1000,
    }


# -------------------------------------------------------------------
# Flask routes
# -------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template(
            "predict.html",
            symptoms_list=symptom_columns or SYMPTOMS_LIST,
            symptoms_display=SYMPTOMS_DISPLAY,
        )

    selected_symptoms = request.form.getlist("symptoms")

    if not selected_symptoms:
        flash(
            "Please select at least one symptom.",
            "warning",
        )
        return redirect(url_for("predict"))

    if model is None or label_encoder is None:
        flash(
            "The prediction model is unavailable. Check the terminal "
            "for the model-loading error, retrain the model, and restart "
            "the Flask app.",
            "danger",
        )
        return redirect(url_for("predict"))

    valid_symptoms = [
        str(symptom).strip().lower()
        for symptom in selected_symptoms
        if str(symptom).strip().lower() in symptom_columns
    ]

    if not valid_symptoms:
        flash(
            "The selected symptoms do not match the trained model.",
            "danger",
        )
        return redirect(url_for("predict"))

    try:
        feature_vector = make_feature_vector(valid_symptoms)

        prediction_value = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]

        predicted_disease = decode_disease_label(
            prediction_value
        )

        top_predictions = get_top_predictions(
            probabilities,
            limit=3,
        )

        confidence = top_predictions[0]["probability"]
        disease_info = get_disease_info(predicted_disease)
        health_advice = get_health_advice(
            predicted_disease,
            disease_info,
        )

        confidence_warning = None

        if confidence < 60:
            confidence_warning = (
                "The model confidence is low. This result is uncertain "
                "and must not be treated as a diagnosis."
            )

        hospital_recommendation = get_hospital_recommendations(
            predicted_disease
        )

        save_prediction(
            valid_symptoms,
            predicted_disease,
            confidence,
        )

        return render_template(
            "result.html",
            predicted_disease=predicted_disease,
            confidence=confidence,
            top_predictions=top_predictions,
            selected_symptoms=valid_symptoms,
            symptoms_display=SYMPTOMS_DISPLAY,
            disease_info=disease_info,
            medicines=health_advice["medicines"],
            precautions=health_advice["precautions"],
            red_flags=health_advice["red_flags"],
            confidence_warning=confidence_warning,
            hospital_recommendation=hospital_recommendation,
            hospital_urgency=hospital_recommendation["urgency"],
        )

    except Exception as exc:
        logger.exception("Prediction error: %s", exc)

        flash(
            "Prediction failed. Check the terminal error and confirm "
            "that the model and encoder were trained and saved together.",
            "danger",
        )
        return redirect(url_for("predict"))


@app.route("/hospitals", methods=["GET", "POST"])
def hospitals():
    results = None
    error = None

    if request.method == "POST":
        location = request.form.get("location", "").strip()
        radius = request.form.get("radius", DEFAULT_RADIUS)

        result = find_nearby_hospitals(
            location_query=location,
            radius=radius,
        )

        if "error" in result:
            error = result["error"]
        else:
            results = result

    return render_template(
        "hospitals.html",
        results=results,
        error=error,
    )


@app.route("/api/hospitals", methods=["POST"])
def hospitals_api():
    data = request.get_json(silent=True) or {}

    result = find_nearby_hospitals(
        location_query=data.get("location"),
        lat=data.get("lat"),
        lon=data.get("lon"),
        radius=data.get("radius", DEFAULT_RADIUS),
    )

    status_code = 200 if "error" not in result else 400
    return result, status_code


@app.route("/history")
def history():
    predictions = get_prediction_history()

    return render_template(
        "history.html",
        predictions=predictions,
        symptoms_display=SYMPTOMS_DISPLAY,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/debug-model")
def debug_model():
    """
    Temporary route for checking whether the model loaded properly.

    Open:
    http://127.0.0.1:5000/debug-model
    """
    return {
        "model_loaded": model is not None,
        "model_type": (
            type(model).__name__
            if model is not None
            else None
        ),
        "label_encoder_loaded": label_encoder is not None,
        "label_encoder_type": (
            type(label_encoder).__name__
            if label_encoder is not None
            else None
        ),
        "loaded_model_path": loaded_model_path,
        "number_of_features": len(symptom_columns),
        "symptom_columns": symptom_columns,
        "model_feature_names": (
            list(model.feature_names_in_)
            if model is not None
            and hasattr(model, "feature_names_in_")
            else None
        ),
        "model_expected_feature_count": (
            int(model.n_features_in_)
            if model is not None
            and hasattr(model, "n_features_in_")
            else None
        ),
    }


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
    )