# AI MediDetect

## AI-Based Disease Prediction and Healthcare Assistance System

**Project By:** Suhani Srivastava  & Krishna Gupta
**Program:** Master of Computer Applications (MCA)  
**Year:** 2024-2026

---

## Quick Start

### Prerequisites
- Python 3.12
- pip (Python package manager)

### Installation (Windows)

1. **Open Command Prompt**
   ```cmd
   cd path\to\AI-MediDetect
   ```

2. **Create Virtual Environment**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Train Model (First Time)**
   ```cmd
   python model\train_model.py
   ```

5. **Run Application**
   ```cmd
   python app.py
   ```

6. **Open Browser**
   Go to: http://127.0.0.1:5000

---

## Features

- ✅ Symptom-based disease prediction
- ✅ Machine learning (Random Forest)
- ✅ 10 diseases, 20 symptoms
- ✅ Hospital finder (OpenStreetMap)
- ✅ Prediction history
- ✅ Responsive design
- ✅ Medical disclaimers

---

## Project Structure

```
AI-MediDetect/
├── app.py                    # Main Flask app
├── requirements.txt          # Dependencies
├── model/
│   ├── train_model.py       # Training script
│   ├── disease_model.pkl    # Trained model
│   └── feature_encoder.pkl  # Encoder
├── data/
│   ├── symptoms.csv         # Dataset
│   ├── diseases.csv         # Disease info
│   └── history.csv          # History
├── templates/               # HTML files
└── static/
    ├── css/style.css
    └── js/script.js
```

---

## Medical Disclaimer

**This is an educational project only.** Not for actual medical diagnosis. Always consult healthcare professionals.

---
## 💊 Medicine Recommendations Feature

AI MediDetect now includes **general medicine recommendations** for each predicted condition.

### What's Included:

- **Common OTC Medicines**: Over-the-counter medications commonly used for symptom relief
- **Important Notes**: Usage instructions and precautions
- **Strong Disclaimers**: Clear warnings about consulting doctors

### Important:

⚠️ **These are NOT prescriptions!** The medicine information is for **educational purposes only**.

- Always consult a qualified healthcare professional before taking any medication
- Dosage depends on individual factors (age, weight, medical history)
- Some medicines may interact with other medications
- Special care needed for pregnant women, children, elderly, and chronic conditions

### Example Output:

For **Common Cold**:
- Paracetamol (for fever/pain)
- Antihistamines (for runny nose)
- Cough syrup (for dry cough)
- Vitamin C supplements
- Saline nasal spray

**Note**: Take paracetamol 500mg every 6 hours if fever. Drink warm water. Avoid antibiotics unless prescribed by doctor.

## Technology Stack

- **Frontend:** HTML5, CSS3, Bootstrap 5, JavaScript
- **Backend:** Python 3.12, Flask
- **ML:** scikit-learn, Pandas, NumPy
- **Maps:** OpenStreetMap, Leaflet.js
- **Deployment:** Gunicorn, Render/Railway

---

## Testing Checklist

- [ ] App starts without errors
- [ ] Homepage loads
- [ ] Symptom selection works
- [ ] Prediction returns results
- [ ] Hospital search works
- [ ] History saves and displays
- [ ] Mobile responsive

---

**Built with ❤️**
