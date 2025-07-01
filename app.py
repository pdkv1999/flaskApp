from flask import Flask, render_template, request, redirect
from flask_pymongo import PyMongo
from datetime import datetime
import joblib
import numpy as np
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# MongoDB Connection
app.config["MONGO_URI"] = "mongodb+srv://124116108:2GX4qsf4uHUGT09e@dileep-nasa-api.xwdqi.mongodb.net/flask_triage_system?retryWrites=true&w=majority&appName=Dileep-NASA-API"
mongo = PyMongo(app)

# === Safe model loading for Replit or Render ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print("Render model directory contents:", os.listdir(MODEL_DIR))  # Debugging

# Load Model and Preprocessing Objects
model = joblib.load(os.path.join(MODEL_DIR, 'adaboost_model.pkl'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'vital_columns.pkl'))

# In-memory queue
queue = []

@app.route('/', methods=['GET'])
def index():
    search_mrn = request.args.get('search_mrn')
    visits = []

    if search_mrn:
        visits = list(mongo.db.visits.find({'mrn': search_mrn}))

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

    # Vectorize complaint
    chiefcomplaint_vector = tfidf.transform([complaint]).toarray()
    vital_input = np.array([[temp, hr, rr, o2, sbp, dbp, 0]])
    final_input = np.hstack((chiefcomplaint_vector, vital_input))
    pred_label = model.predict(final_input)[0]

    color = '🔴' if pred_label == 'Critical' else '🟡' if pred_label == 'Moderate' else '🟢'

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

    # Store to MongoDB
    mongo.db.patients.insert_one(patient)

    # Add to queue
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

            mongo.db.visits.insert_one({
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
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
