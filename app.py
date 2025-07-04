from flask import Flask, render_template, request, redirect, jsonify, flash
from flask_pymongo import PyMongo
from flask_socketio import SocketIO, emit
from datetime import datetime
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'triage_secret_key'

app.config["MONGO_URI"] = "mongodb://localhost:27017/flask_triage_system"
mongo = PyMongo(app)
socketio = SocketIO(app, cors_allowed_origins="*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
model = joblib.load(os.path.join(MODEL_DIR, 'adaboost_model.pkl'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'vital_columns.pkl'))

current_patient = None

@app.route('/')
def index():
    search_mrn = request.args.get('search_mrn')
    visits = list(mongo.db.visits.find({'mrn': search_mrn})) if search_mrn else []

    # Get all booked appointment slots
    appointments = list(mongo.db.patients.find({}, {"_id": 0, "arrival_time": 1}))

    booked_slots = []
    date_counter = {}

    for entry in appointments:
        if "arrival_time" in entry:
            arrival = entry["arrival_time"]
            booked_slots.append(arrival)
            date_part = arrival.split()[0]
            date_counter[date_part] = date_counter.get(date_part, 0) + 1

    # Fully booked dates (16 slots per day)
    fully_booked_days = [date for date, count in date_counter.items() if count >= 16]

    return render_template(
        'patient.html',
        visits=visits if visits else None,
        search=search_mrn,
        booked_slots=booked_slots,
        fully_booked_days=fully_booked_days
    )


@app.route('/submit', methods=['POST'])
def submit():
    mrn = request.form['mrn']
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    complaint = request.form['complaint']
    appointment_date = request.form['appointment_date']
    appointment_time = request.form['appointment_time']
    arrival_time = f"{appointment_date} {appointment_time}"

    sbp, dbp, temp, hr, rr, o2 = map(float, [
        request.form['sbp'],
        request.form['dbp'],
        request.form['temp'],
        request.form['hr'],
        request.form['rr'],
        request.form['o2']
    ])

    # Predict severity
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
        'arrival_time': arrival_time,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    mongo.db.patients.insert_one(patient)
    return redirect('/admin')

@app.route('/admin')
def admin():
    severity_order = {"Critical": 1, "Moderate": 2, "Low": 3}
    patients = list(mongo.db.patients.find())

    # Sort patients by severity then arrival_time
    patients.sort(key=lambda p: (
        severity_order.get(p.get('priority', 'Low'), 4),
        p.get('arrival_time', '')
    ))

    # Group patients by date part of arrival_time
    grouped_patients = {}
    for patient in patients:
        arrival_date = patient.get('arrival_time', '').split(' ')[0]
        grouped_patients.setdefault(arrival_date, []).append(patient)

    # Sort the grouped keys (dates) ascending
    grouped_patients = dict(sorted(grouped_patients.items()))

    return render_template('admin.html', grouped_patients=grouped_patients)


@app.route('/doctor', methods=['GET', 'POST'])
def doctor():
    global current_patient

    if request.method == 'POST' and current_patient:
        medicine = request.form.get('medicine')
        test = request.form.get('test')

        mongo.db.visits.insert_one({
            'mrn': current_patient['mrn'],
            'name': current_patient['name'],
            'complaint': current_patient['complaint'],
            'medicine': medicine,
            'test': test,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # ✅ Handle old data without 'arrival_time'
        delete_query = {'mrn': current_patient['mrn']}
        if 'arrival_time' in current_patient:
            delete_query['arrival_time'] = current_patient['arrival_time']

        mongo.db.patients.delete_one(delete_query)

        current_patient = None
        return redirect('/doctor')

    if not current_patient:
        severity_order = {"Critical": 1, "Moderate": 2, "Low": 3}
        patients = list(mongo.db.patients.find())
        patients.sort(key=lambda p: (
            severity_order.get(p.get('priority', 'Low'), 4),
            p.get('arrival_time', '')
        ))
        if patients:
            current_patient = patients[0]

    return render_template('doctor.html', patient=current_patient)

@app.route('/update_severity', methods=['POST'])
def update_severity():
    data = request.json
    mrn = data.get('mrn')
    new_priority = data.get('priority')

    severity_colors = {'Critical': '🔴', 'Moderate': '🟡', 'Low': '🟢'}
    new_color = severity_colors.get(new_priority, '⚪')

    result = mongo.db.patients.update_one(
        {'mrn': mrn},
        {'$set': {'priority': new_priority, 'color': new_color}}
    )

    socketio.emit('severity_updated', {'mrn': mrn, 'priority': new_priority})
    return jsonify(success=(result.modified_count > 0))

if __name__ == '__main__':
    socketio.run(app, debug=True)
