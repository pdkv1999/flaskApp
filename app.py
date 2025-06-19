from flask import Flask, render_template, request, redirect
from datetime import datetime

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for templates

queue = []
doctor_notes = []

@app.route('/')
def index():
    return render_template('patient.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    complaint = request.form['complaint']
    sbp = request.form['sbp']
    dbp = request.form['dbp']
    temp = request.form['temp']
    hr = request.form['hr']
    rr = request.form['rr']
    o2 = request.form['o2']
    priority = request.form['priority']

    color = '🔴' if priority == 'Critical' else '🟡' if priority == 'Moderate' else '🟢'

    patient = {
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
            patient = queue.pop(0)  # Remove the current patient after treatment
            medicine = request.form.get('medicine')
            test = request.form.get('test')
            doctor_notes.append({
                'patient': patient['name'],
                'medicine': medicine,
                'test': test,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        return redirect('/doctor')  # Show the next patient

    patient = queue[0] if queue else None
    return render_template('doctor.html', patient=patient)

if __name__ == '__main__':
    app.run(debug=True)
