
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import uuid

app = Flask(__name__)

patients_data = []
patient_visits = {}
mrn_map = {}

def generate_mrn():
    return str(uuid.uuid4())[:8]

def get_color(priority):
    return {"Critical": "(Red)", "Moderate": "(Orange)", "Low": "(Green)"}.get(priority, "(Grey)")

def get_time_slot(time_str):
    time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").time()
    if time_obj < datetime.strptime("12:00:00", "%H:%M:%S").time():
        return "morning"
    elif time_obj < datetime.strptime("16:00:00", "%H:%M:%S").time():
        return "afternoon"
    else:
        return "evening"

@app.route('/')
def patient_form():
    return render_template('patient.html', mrn=None, visits=0, prescriptions=[], tests=[], patient=None)

@app.route('/submit', methods=['POST'])
def submit_patient():
    name = request.form['name']
    mrn = request.form.get('mrn') or generate_mrn()
    mrn_map[name] = mrn
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    visit_data = {
        'name': name,
        'age': request.form['age'],
        'gender': request.form['gender'],
        'complaint': request.form['complaint'],
        'sbp': request.form['sbp'],
        'dbp': request.form['dbp'],
        'temp': request.form['temp'],
        'hr': request.form['hr'],
        'rr': request.form['rr'],
        'o2': request.form['o2'],
        'priority': request.form['priority'],
        'time': now,
        'mrn': mrn,
        'color': get_color(request.form['priority']),
        'prescriptions': [],
        'tests': []
    }

    patients_data.append(visit_data)

    if mrn not in patient_visits:
        patient_visits[mrn] = []
    patient_visits[mrn].append(visit_data)

    return render_template('patient.html',
                           mrn=mrn,
                           visits=len(patient_visits[mrn]),
                           prescriptions=visit_data['prescriptions'],
                           tests=visit_data['tests'],
                           patient=visit_data)

@app.route('/lookup', methods=['POST'])
def lookup_patient():
    mrn = request.form['mrn']
    if mrn in patient_visits:
        last_visit = patient_visits[mrn][-1]
        return render_template('patient.html',
                               mrn=mrn,
                               visits=len(patient_visits[mrn]),
                               prescriptions=last_visit['prescriptions'],
                               tests=last_visit['tests'],
                               patient=last_visit)
    return render_template('patient.html', mrn=None, visits=0, prescriptions=[], tests=[], patient=None)

@app.route('/admin')
def admin_dashboard():
    grouped = {"morning": [], "afternoon": [], "evening": []}
    for patient in patients_data:
        slot = get_time_slot(patient['time'])
        grouped[slot].append(patient)
    for slot in grouped:
        grouped[slot].sort(key=lambda x: {"Critical": 1, "Moderate": 2, "Low": 3}.get(x['priority'], 4))
    return render_template('admin.html',
                           morning=grouped["morning"],
                           afternoon=grouped["afternoon"],
                           evening=grouped["evening"])

@app.route('/doctor', methods=['GET', 'POST'])
def doctor_view():
    if request.method == 'POST':
        medicine = request.form.get('medicine')
        test = request.form.get('test')
        if patients_data:
            patient = patients_data[0]
            mrn = patient['mrn']
            if patient in patient_visits.get(mrn, []):
                if medicine:
                    patient['prescriptions'].append(medicine)
                if test:
                    patient['tests'].append(test)
            patients_data.pop(0)
        return redirect(url_for('doctor_view'))
    return render_template('doctor.html', patient=patients_data[0] if patients_data else None)

if __name__ == '__main__':
    app.run(debug=True)
