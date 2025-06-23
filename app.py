from flask import Flask, render_template, request, redirect
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Load AdaBoost model and metadata
model = joblib.load('models/adaboost_model.pkl')
feature_columns = joblib.load('models/vital_columns.pkl')
label_encoder = joblib.load('models/tfidf_vectorizer.pkl')

queue = []
doctor_notes = []

@app.route('/', methods=['GET'])
def index():
    search_mrn = request.args.get('search_mrn')
    visits = []
    
    if search_mrn:
        # Combine queue and doctor_notes for complete visit history
        for entry in doctor_notes:
            if entry['mrn'] == search_mrn:
                visits.append(entry)
    
    return render_template('patient.html', visits=visits if visits else None, search=search_mrn)

@app.route('/submit', methods=['POST'])
def submit():
    mrn = request.form['mrn']
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    complaint = request.form['complaint']
    sbp = float(request.form['sbp'])
    dbp = float(request.form['dbp'])
    temp = float(request.form['temp'])
    hr = float(request.form['hr'])
    rr = float(request.form['rr'])
    o2 = float(request.form['o2'])

    # Map complaint to encoded value used in training
    complaint_mapping = {
        'fever': 1,
        'chest pain': 2,
        'headache': 3,
        'abdominal pain': 4,
        'cough': 5
    }
    chiefcomplaint = complaint_mapping.get(complaint.lower(), 0)

    # Build model input
    input_dict = {
        'chiefcomplaint': chiefcomplaint,
        'temperature': temp,
        'heartrate': hr,
        'resprate': rr,
        'o2sat': o2,
        'sbp': sbp,
        'dbp': dbp,
        'pain': 0  # You can make this dynamic later
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    # Predict
    pred_encoded = model.predict(input_df)[0]
    priority = label_encoder.inverse_transform([pred_encoded])[0]

    color = '🔴' if priority == 'Critical' else '🟡' if priority == 'Moderate' else '🟢'

    patient = {
        'mrn': mrn,
        'name': name,
        'age': age,
        'gender': gender,
        'complaint': complaint,
        'sbp': sbp,
        'dbp': dbp,
        'temp': temp,
        'hr': hr,
        'rr': rr,
        'o2': o2,
        'priority': priority,
        'color': color,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    queue.append(patient)
    queue.sort(key=lambda x: {'Critical': 1, 'Moderate': 2, 'Low': 3}[x['priority']])
    return redirect('/admin')

@app.route('/admin')
def admin():
    return render_template('admin.html', patients=queue)

@app.route('/doctor', methods=['GET', 'POST'])
def doctor():
    if request.method == 'POST':
        if queue:
            patient = queue.pop(0)  # Remove treated patient
            medicine = request.form.get('medicine')
            test = request.form.get('test')
            doctor_notes.append({
                'mrn': patient['mrn'],
                'name': patient['name'],
                'complaint': patient['complaint'],
                'medicine': medicine,
                'test': test,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        return redirect('/doctor')

    patient = queue[0] if queue else None
    return render_template('doctor.html', patient=patient)

if __name__ == '__main__':
    app.run(debug=True)
