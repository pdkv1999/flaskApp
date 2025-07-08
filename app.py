from flask import Flask, render_template, request, redirect, jsonify, flash
from flask_pymongo import PyMongo
from flask_socketio import SocketIO, emit
from datetime import datetime
import joblib
import numpy as np
import os
from collections import Counter, defaultdict
from flask import session

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
    if 9 <= hour < 13:
        return 'Morning (9-1)'
    elif 13 <= hour < 17:
        return 'Afternoon (1-5)'
    elif 17 <= hour < 21:
        return 'Evening (5-9)'
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

from flask import render_template, request
from datetime import datetime

@app.route('/admin')
def admin():
    severity_order = {"Critical": 1, "Moderate": 2, "Low": 3}
    patients = list(mongo.db.patients.find())

    patients.sort(key=lambda p: (
        p.get('arrival_time', '')[:10],  # Date part: YYYY-MM-DD
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

    time_block_order = ['Morning (9-1)', 'Afternoon (1-5)', 'Evening (5-9)', 'Late Night (9-12)', 'Other']
    grouped_patients = dict(sorted(grouped_patients.items()))
    for date in grouped_patients:
        grouped_patients[date] = dict(sorted(
            grouped_patients[date].items(),
            key=lambda x: time_block_order.index(x[0]) if x[0] in time_block_order else len(time_block_order)
        ))

    # Ensure all weekdays are represented
    all_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in all_weekdays:
        if day not in weekday_severity_data:
            weekday_severity_data[day] = {'Critical': 0, 'Moderate': 0, 'Low': 0}
        if day not in weekday_patient_counts:
            weekday_patient_counts[day] = 0

    # Default peak hour and weekday
    peak_time = time_frequencies.most_common(1)[0][0] if time_frequencies else None
    default_peak_day = weekday_patient_counts.most_common(1)[0][0] if weekday_patient_counts else "Monday"

    # ✅ Add pagination logic (safe extension)
    date_keys = list(grouped_patients.keys())
    selected_date = request.args.get('date', date_keys[0] if date_keys else None)

    prev_date = next_date = None
    if selected_date in date_keys:
        current_index = date_keys.index(selected_date)
        if current_index > 0:
            prev_date = date_keys[current_index - 1]
        if current_index < len(date_keys) - 1:
            next_date = date_keys[current_index + 1]
    else:
        selected_date = None

    return render_template(
        'admin.html',
        grouped_patients=grouped_patients,
        severity_counts=severity_counts,
        peak_time=peak_time,
        date_to_weekday_map=date_to_weekday_map,
        weekday_severity_data=dict(weekday_severity_data),
        weekday_patient_counts=dict(weekday_patient_counts),
        default_peak_day=default_peak_day,

        # New context for pagination-based view
        selected_date=selected_date,
        selected_day=datetime.strptime(selected_date, "%Y-%m-%d").strftime("%A") if selected_date else None,
        patients_by_block=grouped_patients.get(selected_date, {}),
        prev_date=prev_date,
        next_date=next_date
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
    if request.method == 'POST':
        patient = session.get('current_patient')
        if patient:
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

            mongo.db.patients.delete_one({
                'mrn': patient['mrn'],
                'arrival_time': patient['arrival_time']
            })

            session.pop('patient_index', None)
            session.pop('current_patient', None)
            return redirect('/doctor')

    time_block_order = ['Morning (9-1)', 'Afternoon (1-5)', 'Evening (5-9)', 'Late Night (9-12)', 'Other']
    severity_order = {"Critical": 1, "Moderate": 2, "Low": 3}

    patients = list(mongo.db.patients.find())

    def sort_key(p):
        arrival = p.get('arrival_time', '')
        if not arrival:
            return ('9999-12-31', len(time_block_order), 99, '9999-12-31 23:59:59')

        date_part = arrival[:10]
        time_block = get_time_block(arrival)
        time_block_index = time_block_order.index(time_block) if time_block in time_block_order else len(time_block_order)
        severity = severity_order.get(p.get('priority', 'Low'), 4)
        return (date_part, time_block_index, severity, arrival)

    patients.sort(key=sort_key)

    index = session.get('patient_index', 0)

    if request.args.get('next') == 'true':
        index += 1
        if index >= len(patients):
            index = 0
        session['patient_index'] = index

    if patients:
        current = patients[index]
        session['current_patient'] = current
        remaining_count = len(patients) - index
    else:
        current = None
        session.pop('patient_index', None)
        session.pop('current_patient', None)
        remaining_count = 0  # 🔧 Define it even if no patients

    return render_template('doctor.html', patient=current, remaining_count=remaining_count)

@app.route('/update_severity', methods=['POST'])
def update_severity():
    data = request.json
    mrn = data.get('mrn')
    arrival_time = data.get('arrival_time')
    new_priority = data.get('priority')

    severity_colors = {'Critical': '🔴', 'Moderate': '🟡', 'Low': '🟢'}
    new_color = severity_colors.get(new_priority, '⚪')

    result = mongo.db.patients.update_one(
        {'mrn': mrn, 'arrival_time': arrival_time},
        {'$set': {'priority': new_priority, 'color': new_color}}
    )

    return jsonify(success=(result.modified_count > 0))

if __name__ == '__main__':
    socketio.run(app, debug=True)
