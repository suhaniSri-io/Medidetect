"""
AI MediDetect - Main Flask Application
AI-Based Disease Prediction and Healthcare Assistance System
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os
import json

app = Flask(__name__)
app.secret_key = 'ai-medidetect-secret-key-2024'

SYMPTOMS_LIST = [
    'fever', 'headache', 'cough', 'fatigue', 'nausea', 'vomiting',
    'sore_throat', 'body_pain', 'stomach_pain', 'chest_pain',
    'shortness_of_breath', 'dizziness', 'joint_pain', 'skin_rash',
    'runny_nose', 'chills', 'abdominal_pain', 'loss_of_appetite',
    'diarrhea', 'muscle_pain'
]

SYMPTOMS_DISPLAY = {
    'fever': 'Fever', 'headache': 'Headache', 'cough': 'Cough', 'fatigue': 'Fatigue',
    'nausea': 'Nausea', 'vomiting': 'Vomiting', 'sore_throat': 'Sore Throat',
    'body_pain': 'Body Pain', 'stomach_pain': 'Stomach Pain', 'chest_pain': 'Chest Pain',
    'shortness_of_breath': 'Shortness of Breath', 'dizziness': 'Dizziness',
    'joint_pain': 'Joint Pain', 'skin_rash': 'Skin Rash', 'runny_nose': 'Runny Nose',
    'chills': 'Chills', 'abdominal_pain': 'Abdominal Pain', 'loss_of_appetite': 'Loss of Appetite',
    'diarrhea': 'Diarrhea', 'muscle_pain': 'Muscle Pain'
}

MODEL_PATH = os.path.join('model', 'disease_model.pkl')
ENCODER_PATH = os.path.join('model', 'feature_encoder.pkl')
DISEASES_PATH = os.path.join('data', 'diseases.csv')
HISTORY_PATH = os.path.join('data', 'history.csv')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder_data = pickle.load(f)
    label_encoder = encoder_data['label_encoder']
    symptom_columns = encoder_data['symptom_columns']
    print("Model and encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoder = None
    symptom_columns = SYMPTOMS_LIST

try:
    diseases_df = pd.read_csv(DISEASES_PATH)
    diseases_info = diseases_df.set_index('disease').to_dict('index')
    print("Diseases information loaded!")
except Exception as e:
    print(f"Error loading diseases info: {e}")
    diseases_info = {}


def get_disease_info(disease_name):
    if disease_name in diseases_info:
        return diseases_info[disease_name]
    return None


def save_prediction(symptoms, predicted_disease, confidence):
    try:
        history_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symptoms': ','.join(symptoms),
            'predicted_disease': predicted_disease,
            'confidence': f"{confidence:.1f}%"
        }
        if os.path.exists(HISTORY_PATH):
            history_df = pd.read_csv(HISTORY_PATH)
            history_df = pd.concat([history_df, pd.DataFrame([history_data])], ignore_index=True)
        else:
            history_df = pd.DataFrame([history_data])
        history_df.to_csv(HISTORY_PATH, index=False)
        return True
    except Exception as e:
        print(f"Error saving history: {e}")
        return False


def get_prediction_history():
    try:
        if os.path.exists(HISTORY_PATH):
            df = pd.read_csv(HISTORY_PATH)
            return df.to_dict('records')
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []


def find_nearby_hospitals(location_query=None, lat=None, lon=None, radius=5000):
    """Find nearby hospitals using OpenStreetMap Overpass API with better error handling"""
    hospitals = []
    
    try:
        # If location query provided, geocode it first using Nominatim
        if location_query and location_query.strip() and not (lat and lon):
            print(f"Geocoding location: {location_query}")
            
            nominatim_url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': location_query,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {
                'User-Agent': 'AIMediDetect/1.0 (Educational Healthcare Project)',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            try:
                response = requests.get(nominatim_url, params=params, headers=headers, timeout=15)
                print(f"Nominatim response status: {response.status_code}")
                
                if response.status_code == 200:
                    results = response.json()
                    if results and len(results) > 0:
                        result = results[0]
                        lat = float(result['lat'])
                        lon = float(result['lon'])
                        location_name = result.get('display_name', location_query)
                        print(f"Location found: {location_name} ({lat}, {lon})")
                    else:
                        return {'error': f"Location '{location_query}' not found. Please try a different city name or use coordinates."}
                else:
                    return {'error': f"Geocoding service temporarily unavailable (Status: {response.status_code})"}
            except requests.exceptions.RequestException as e:
                return {'error': f'Unable to connect to location service. Please check your internet connection. Error: {str(e)}'}
        
        # Validate we have coordinates
        if not (lat and lon):
            return {'error': 'Please provide a valid location name or latitude/longitude coordinates'}
        
        # Validate coordinates are numbers
        try:
            lat = float(lat)
            lon = float(lon)
            
            # Basic coordinate validation
            if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                return {'error': 'Invalid coordinates. Latitude must be between -90 and 90, Longitude between -180 and 180'}
        except (ValueError, TypeError):
            return {'error': 'Invalid coordinates. Please enter valid numbers for latitude and longitude'}
        
        # Query Overpass API for hospitals
        print(f"Searching hospitals around ({lat}, {lon}) within {radius}m radius")
        
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][30];
        (
          node["amenity"="hospital"](around:{radius},{lat},{lon});
          way["amenity"="hospital"](around:{radius},{lat},{lon});
          relation["amenity"="hospital"](around:{radius},{lat},{lon});
          node["amenity"="clinic"](around:{radius},{lat},{lon});
          way["amenity"="clinic"](around:{radius},{lat},{lon});
        );
        out body center;
        """
        
        try:
            response = requests.post(overpass_url, data={'data': overpass_query}, timeout=45)
            print(f"Overpass API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                elements = data.get('elements', [])
                print(f"Found {len(elements)} hospitals/clinics")
                
                for element in elements:
                    tags = element.get('tags', {})
                    
                    # Get hospital name
                    hospital_name = tags.get('name', tags.get('operator', 'Unknown Hospital'))
                    
                    # Get address information
                    address_parts = []
                    if tags.get('addr:full'):
                        address_parts.append(tags.get('addr:full'))
                    elif tags.get('addr:street'):
                        street = tags.get('addr:street', '')
                        housenumber = tags.get('addr:housenumber', '')
                        if housenumber:
                            street = f"{housenumber} {street}"
                        address_parts.append(street)
                        if tags.get('addr:city'):
                            address_parts.append(tags.get('addr:city'))
                        if tags.get('addr:state'):
                            address_parts.append(tags.get('addr:state'))
                    elif tags.get('address'):
                        address_parts.append(tags.get('address'))
                    else:
                        # Try to get city from element or use search location
                        if location_query:
                            address_parts.append(f"Near {location_query}")
                        else:
                            address_parts.append("Address not available")
                    
                    address = ', '.join(address_parts) if address_parts else "Address not available"
                    
                    # Get coordinates
                    if 'center' in element:
                        h_lat = element['center']['lat']
                        h_lon = element['center']['lon']
                    else:
                        h_lat = element.get('lat', lat)
                        h_lon = element.get('lon', lon)
                    
                    # Calculate approximate distance
                    # Using Haversine formula for better accuracy
                    from math import radians, sin, cos, sqrt, atan2
                    
                    R = 6371  # Earth's radius in km
                    
                    lat1, lon1 = radians(lat), radians(lon)
                    lat2, lon2 = radians(h_lat), radians(h_lon)
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    
                    distance_km = R * c
                    
                    # Get phone and emergency info
                    phone = tags.get('phone', tags.get('contact:phone', ''))
                    emergency = tags.get('emergency', 'no') == 'yes'
                    website = tags.get('website', tags.get('contact:website', ''))
                    
                    hospital = {
                        'name': hospital_name,
                        'lat': h_lat,
                        'lon': h_lon,
                        'address': address,
                        'phone': phone,
                        'emergency': emergency,
                        'website': website,
                        'distance': f"{distance_km:.2f} km"
                    }
                    
                    hospitals.append(hospital)
                
                if not hospitals:
                    return {
                        'error': f'No hospitals or clinics found within {radius/1000:.1f} km of {location_query or f"({lat}, {lon})"}. Try increasing the search radius or checking a different location.',
                        'suggestion': 'Try searching for a major city or use a larger radius (10-20 km)'
                    }
                
                # Sort by distance
                hospitals.sort(key=lambda x: float(x['distance'].split()[0]))
                
                return {
                    'hospitals': hospitals,
                    'center': {'lat': lat, 'lon': lon},
                    'count': len(hospitals),
                    'search_location': location_query or f"({lat}, {lon})",
                    'radius_km': radius / 1000
                }
            else:
                return {'error': f'Hospital search service temporarily unavailable (Status: {response.status_code}). Please try again later.'}
        
        except requests.exceptions.Timeout:
            return {'error': 'Hospital search timed out. Please try again with a smaller radius or check your internet connection.'}
        except requests.exceptions.RequestException as e:
            return {'error': f'Unable to connect to hospital search service. Please check your internet connection. Error: {str(e)}'}
        except Exception as e:
            return {'error': f'Error processing hospital data: {str(e)}'}
    
    except Exception as e:
        return {'error': f'Unexpected error occurred: {str(e)}. Please try again or contact support.'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', symptoms_list=SYMPTOMS_LIST, symptoms_display=SYMPTOMS_DISPLAY)

    selected_symptoms = request.form.getlist('symptoms')

    if not selected_symptoms:
        flash('Please select at least one symptom.', 'warning')
        return redirect(url_for('predict'))

    valid_symptoms = [s for s in selected_symptoms if s in SYMPTOMS_LIST]
    if not valid_symptoms:
        flash('Invalid symptoms selected.', 'danger')
        return redirect(url_for('predict'))

    feature_vector = np.zeros(len(symptom_columns))
    for symptom in valid_symptoms:
        if symptom in symptom_columns:
            idx = symptom_columns.index(symptom)
            feature_vector[idx] = 1

    try:
        prediction = model.predict([feature_vector])[0]
        probabilities = model.predict_proba([feature_vector])[0]
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100
        disease_info = get_disease_info(predicted_disease)
        save_prediction(valid_symptoms, predicted_disease, confidence)

        return render_template('result.html', predicted_disease=predicted_disease, confidence=confidence,
                               selected_symptoms=valid_symptoms, symptoms_display=SYMPTOMS_DISPLAY,
                               disease_info=disease_info)
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'danger')
        return redirect(url_for('predict'))


@app.route('/hospitals', methods=['GET', 'POST'])
def hospitals():
    results = None
    error = None

    if request.method == 'POST':
        location = request.form.get('location', '').strip()
        lat = request.form.get('lat', '').strip()
        lon = request.form.get('lon', '').strip()
        radius = request.form.get('radius', '5000')

        try:
            radius = int(radius)
        except ValueError:
            radius = 5000

        if not location and not (lat and lon):
            error = 'Please enter a location or coordinates'
        else:
            result = find_nearby_hospitals(
                location_query=location if location else None,
                lat=float(lat) if lat else None,
                lon=float(lon) if lon else None,
                radius=radius
            )
            if 'error' in result:
                error = result['error']
            else:
                results = result

    return render_template('hospitals.html', results=results, error=error)


@app.route('/history')
def history():
    predictions = get_prediction_history()
    return render_template('history.html', predictions=predictions, symptoms_display=SYMPTOMS_DISPLAY)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
