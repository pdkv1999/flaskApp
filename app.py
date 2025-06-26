from flask import Flask, render_template, request, redirect
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Load Model and Preprocessing Objects
model = joblib.load('models/adaboost_model.pkl')           # Trained AdaBoost model
tfidf = joblib.load('models/tfidf_vectorizer.pkl')         # Trained TF-IDF vectorizer
feature_columns = joblib.load('models/vital_columns.pkl')  # Vital columns order (optional)

# Queue and visit history
queue = []
doctor_notes = []

@app.route('/', methods=['GET'])
def index():
    search_mrn = request.args.get('search_mrn')
    visits = []

    if search_mrn:
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

    # Step 1: Vectorize chief complaint using TF-IDF
    chiefcomplaint_vector = tfidf.transform([complaint]).toarray()

    # Step 2: Prepare numeric vital features
    vital_input = np.array([[temp, hr, rr, o2, sbp, dbp, 0]])  # Pain is 0 (placeholder)

    # Step 3: Combine TF-IDF and vitals
    final_input = np.hstack((chiefcomplaint_vector, vital_input))

    # Step 4: Predict
    pred_label = model.predict(final_input)[0]  # Returns "Critical", "Moderate", or "Low"

    # Priority color
    color = '🔴' if pred_label == 'Critical' else '🟡' if pred_label == 'Moderate' else '🟢'

    # Step 5: Build patient object
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
        'priority': pred_label,
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
            patient = queue.pop(0)
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
