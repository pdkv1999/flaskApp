# 🏥 AI-Enhanced Hospital Triage System

A full-stack Flask application that streamlines emergency department triage using intelligent priority classification, real-time patient queue management, and a role-based interface for **patients**, **admins**, and **doctors**.

---

## 🚀 Features

### 🔷 Patient Registration
- Patients can register via a clean and responsive web form.
- Default vitals (SBP, DBP, Temp, HR, RR, O2) are auto-filled for convenience.
- Real-time **MRN-based auto-fill** of patient demographics if a matching record exists.
- Flash warnings and validation for incomplete/invalid entries (e.g., invalid MRN).

### 🤖 AI-Powered Triage Prediction
- On submission, the system:
  - Processes vital signs and chief complaint text.
  - Feeds the input to a **trained AdaBoost classifier**.
  - Assigns a severity category: **Critical 🔴**, **Moderate 🟡**, or **Low 🟢**.
- Patients are then placed in a MongoDB-based queue, prioritized by:
  1. **Date**
  2. **Time block** (Morning → Late Night)
  3. **Severity**

### 🧑‍⚕️ Doctor Dashboard
- Displays one patient at a time, with:
  - MRN
  - Name
  - Predicted Severity
- Doctors can:
  - Enter prescribed medicines and ordered tests.
  - Submit completed visits, which are stored in a **`visits`** collection.
- Shows **number of patients remaining** in queue.
- Supports real-time **Socket.IO notifications** when severity is updated by an admin.

### 🛠 Admin Dashboard
- View and manage triaged patients by:
  - **Date**
  - **Time block** (Morning, Afternoon, Evening, Late Night)
  - **Severity group**
- Editable severity via dropdown; changes are immediately persisted and broadcasted to doctor screen.
- Includes:
  - Pagination between appointment dates.
  - Severity summary by **weekday** using a **Chart.js bar graph**.
  - Interactive weekday selection showing patient volume.

### 📊 Patient Visit History
- Users can search by MRN to see:
  - All previous visits.
  - Complaints, test orders, and medicines prescribed.
- Uses Bootstrap collapsible cards for better readability.

---

## 🧠 Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Flask, Flask-SocketIO, Flask-PyMongo |
| **Database** | MongoDB |
| **Machine Learning** | AdaBoostClassifier (trained via scikit-learn), TF-IDF for text |
| **Frontend** | HTML5, CSS3, Bootstrap 5, Jinja2 |
| **Real-Time** | WebSockets (Socket.IO) |
| **Charting** | Chart.js |

---

## 📂 Folder Structure

```
project/
│
├── templates/
│   ├── patient.html         # Registration and visit search
│   ├── doctor.html          # Doctor dashboard
│   ├── admin.html           # Admin dashboard
│
├── static/
│   └── images/hospital_bg.jpg
│
├── models/
│   ├── adaboost_model.pkl   # Trained ML model
│   ├── tfidf_vectorizer.pkl
│   └── vital_columns.pkl
│
├── app.py                   # Main Flask application
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone git@github.com:pdkv1999/flaskApp.git
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure MongoDB is running locally**
   - Database: `flask_triage_system`
   - Collections: `patients`, `visits`

4. **Run the Flask app**
   ```bash
   python app.py
   ```

5. **Access in browser**
   ```
   http://localhost:5000
   ```

---

## 🔒 Access Roles

| Role    | Description |
|---------|-------------|
| **Patient** | Submits vitals & complaints |
| **Admin**   | Manages severity & triage queue |
| **Doctor**  | Views next patient, records visit |

---

## 📈 ML Details

- The AdaBoost model classifies patient severity using:
  - Chief complaint (vectorized using TF-IDF)
  - Vitals: temperature, heart rate, respiratory rate, oxygen, systolic BP, diastolic BP
- Trained offline and loaded via `joblib`.

---
