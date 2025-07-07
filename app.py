from flask import Flask, render_template, request, redirect, jsonify, flash
from flask_pymongo import PyMongo
from flask_socketio import SocketIO, emit
from datetime import datetime
import joblib
import numpy as np
import os
from collections import Counter, defaultdict

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

def get_time_block(arrival_str):
    try:
        time_part = arrival_str.split()[1]
        hour = int(time_part.split(':')[0])
    except Exception:
        return 'Other'
    if 9 <= hour < 12:
        return 'Morning (9-12)'
    elif 13 <= hour < 16:
        return 'Afternoon (1-4)'
    elif 17 <= hour < 20:
        return 'Evening (5-8)'
    elif 21 <= hour < 24:
        return 'Late Night (9-12)'
    else:
        return 'Other'

@app.route('/')
def index():
    # (unchanged index logic)
    search_mrn = request.args.get('search_mrn')
    visits = list(mongo.db.visits.find({'mrn': search_mrn})) if search_mrn else []

    patient_name = None
    patient_mrn = None

    if visits:
        patient_info = mongo.db.patients.find_one({'mrn': search_mrn})
        if patient_info:
            patient_name = patient_info.get('name', 'N/A')
            patient_mrn = patient_info.get('mrn', 'N/A')
        else:
            patient_name = visits[0].get('name', 'N/A')
            patient_mrn = search_mrn

    appointments = list(mongo.db.patients.find({}, {"_id": 0, "arrival_time": 1}))
    booked_slots = []
    date_counter = {}

    for entry in appointments:
        if "arrival_time" in entry:
            arrival = entry["arrival_time"]
            booked_slots.append(arrival)
            date_part = arrival.split()[0]
            date_counter[date_part] = date_counter.get(date_part, 0) + 1

    fully_booked_days = [date for date, count in date_counter.items() if count >= 16]

    return render_template(
        'patient.html',
        visits=visits if visits else None,
        search=search_mrn,
        patient_name=patient_name,
        patient_mrn=patient_mrn,
        booked_slots=booked_slots,
        fully_booked_days=fully_booked_days
    )

@app.route('/admin')
def admin():
    severity_order = {"Critical": 1, "Moderate": 2, "Low": 3}
    patients = list(mongo.db.patients.find())

    patients.sort(key=lambda p: (
        severity_order.get(p.get('priority', 'Low'), 4),
        p.get('arrival_time', '')
    ))

    grouped_patients = {}
    severity_counts = Counter()
    time_frequencies = Counter()
    date_to_weekday_map = {}
    weekday_severity_data = defaultdict(lambda: {'Critical': 0, 'Moderate': 0, 'Low': 0})
    weekday_patient_counts = Counter()

    for patient in patients:
        arrival_time = patient.get('arrival_time', '')
        if not arrival_time:
            continue
        arrival_date = arrival_time.split(' ')[0]
        arrival_hour_min = arrival_time.split(' ')[1]

        weekday = datetime.strptime(arrival_date, "%Y-%m-%d").strftime("%A")
        weekday_patient_counts[weekday] += 1
        weekday_severity_data[weekday][patient.get('priority', 'Low')] += 1

        time_block = get_time_block(arrival_time)

        if arrival_date not in grouped_patients:
            grouped_patients[arrival_date] = {}
        grouped_patients[arrival_date].setdefault(time_block, []).append(patient)

        severity_counts[patient.get('priority', 'Low')] += 1
        time_frequencies[arrival_hour_min] += 1

        if arrival_date not in date_to_weekday_map:
            date_to_weekday_map[arrival_date] = weekday

    time_block_order = ['Morning (9-12)', 'Afternoon (1-4)', 'Evening (5-8)', 'Late Night (9-12)', 'Other']
    grouped_patients = dict(sorted(grouped_patients.items()))
    for date in grouped_patients:
        grouped_patients[date] = dict(sorted(grouped_patients[date].items(), key=lambda x: time_block_order.index(x[0])))

    peak_time = time_frequencies.most_common(1)[0][0] if time_frequencies else None
    default_peak_day = weekday_patient_counts.most_common(1)[0][0] if weekday_patient_counts else None

    return render_template(
        'admin.html',
        grouped_patients=grouped_patients,
        severity_counts=severity_counts,
        peak_time=peak_time,
        date_to_weekday_map=date_to_weekday_map,
        weekday_severity_data=dict(weekday_severity_data),
        weekday_patient_counts=dict(weekday_patient_counts),
        default_peak_day=default_peak_day
    )


@app.route('/get_patient_info')
def get_patient_info():
    # (unchanged)
    mrn = request.args.get('mrn', '').strip()
    if not mrn:
        return jsonify({'error': 'MRN required'}), 400
    
    patient = mongo.db.patients.find_one({'mrn': mrn})
    if patient:
        return jsonify({
            'name': patient.get('name', ''),
            'age': patient.get('age', ''),
            'gender': patient.get('gender', '')
        })
    else:
        return jsonify({})

@app.route('/submit', methods=['POST'])
def submit():
    # (unchanged submit logic)
    mrn = request.form['mrn']
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    complaint = request.form['complaint']

    arrival_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    sbp, dbp, temp, hr, rr, o2 = map(float, [
        request.form['sbp'],
        request.form['dbp'],
        request.form['temp'],
        request.form['hr'],
        request.form['rr'],
        request.form['o2']
    ])

    chiefcomplaint_vector = tfidf.transform([complaint]).toarray()
    vital_input = np.array([[temp, hr, rr, o2, sbp, dbp, 0]])
    final_input = np.hstack((chiefcomplaint_vector, vital_input))

    try:
        probs = model.predict_proba(final_input)[0]
    except AttributeError:
        probs = []

    if len(probs) == 3:
        critical_prob, low_prob, moderate_prob = probs
        if critical_prob > 0.5:
            pred_label = 'Critical'
        else:
            if low_prob >= moderate_prob:
                pred_label = 'Low'
            else:
                pred_label = 'Low'
    else:
        pred_label = 'Low'

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
        'time': arrival_time
    }

    mongo.db.patients.insert_one(patient)
    return redirect('/admin')

@app.route('/doctor', methods=['GET', 'POST'])
def doctor():
    # (unchanged)
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
    # (unchanged)
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
