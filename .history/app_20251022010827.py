"""
AnemoCheck - Anemia Classification Web Application
-------------------------------------------------
This Flask application provides a web interface for the anemia classification system.
It features user authentication, real-time updates, and a comprehensive admin dashboard.

Date: April 28, 2025
"""

import os
import json
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, FloatField, TextAreaField, BooleanField, SelectField, SubmitField, HiddenField, IntegerField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import io
import base64

import database as db
import simple_chat

from anemia_model import AnemiaCBCModel
import joblib
from xgboost_ml_module import xgboost_predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'a-very-secret-key-for-anemocheck'
csrf = CSRFProtect(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize anemia model
anemia_model = AnemiaCBCModel()


# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = user_data['id']
        self.username = user_data['username']
        self.email = user_data['email']
        self.first_name = user_data['first_name']
        self.last_name = user_data['last_name']
        self.gender = user_data['gender']
        self.date_of_birth = user_data['date_of_birth']
        self.medical_id = user_data['medical_id']
        self.is_admin = user_data['is_admin']
        self.created_at = user_data['created_at']
        self.last_login = user_data['last_login']


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    user_data = db.get_user(user_id)
    if user_data:
        return User(user_data)
    return None


# Form classes
class LoginForm(FlaskForm):
    """Login form."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    """Registration form."""
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    first_name = StringField('First Name', validators=[Length(max=64)])
    last_name = StringField('Last Name', validators=[Length(max=64)])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')], validators=[DataRequired()])
    date_of_birth = StringField('Date of Birth (DD-MM-YYYY)', validators=[Optional()])
    medical_id = StringField('Medical ID (Optional)', validators=[Optional(), Length(max=64)])
    submit = SubmitField('Register')


class ProfileForm(FlaskForm):
    """Form for updating user profile."""
    first_name = StringField('First Name', validators=[Length(max=64)])
    last_name = StringField('Last Name', validators=[Length(max=64)])
    username = StringField('Username', validators=[Length(max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')])
    date_of_birth = StringField('Date of Birth (DD-MM-YYYY)', validators=[Optional()])
    medical_id = StringField('Medical ID', validators=[Optional(), Length(max=64)])
    current_password = PasswordField('Current Password', validators=[Optional()])
    new_password = PasswordField('New Password', validators=[Optional(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password', validators=[EqualTo('new_password')])
    submit = SubmitField('Update Profilaze')


class MedicalDataForm(FlaskForm):
    """Form for updating medical data."""
    height = FloatField('Height (cm)', validators=[Optional(), NumberRange(min=50, max=250)])
    weight = FloatField('Weight (kg)', validators=[Optional(), NumberRange(min=1, max=500)])
    blood_type = SelectField('Blood Type', choices=[
        ('', 'Unknown'),
        ('A+', 'A+'), ('A-', 'A-'),
        ('B+', 'B+'), ('B-', 'B-'),
        ('AB+', 'AB+'), ('AB-', 'AB-'),
        ('O+', 'O+'), ('O-', 'O-')
    ], validators=[Optional()])
    medical_conditions = TextAreaField('Medical Conditions', validators=[Optional(), Length(max=1000)])
    medications = TextAreaField('Current Medications', validators=[Optional(), Length(max=1000)])
    submit = SubmitField('Update Medical Data')


class CBCForm(FlaskForm):
    """Form for CBC data input."""
    hemoglobin = FloatField('Hemoglobin (g/dL)', validators=[
        DataRequired(), 
        NumberRange(min=1, max=25, message='Please enter a valid value between 1 and 25')
    ])
    notes = TextAreaField('Notes', validators=[Optional(), Length(max=500)])
    submit = SubmitField('Detect Anemia')


class AdminUserForm(FlaskForm):
    """Form for admin to edit user data."""
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    first_name = StringField('First Name', validators=[Length(max=64)])
    last_name = StringField('Last Name', validators=[Length(max=64)])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')])
    date_of_birth = StringField('Date of Birth (DD-MM-YYYY)', validators=[Optional(), Length(max=10)])
    medical_id = StringField('Medical ID', validators=[Optional(), Length(max=64)])
    is_admin = BooleanField('Administrator')
    password = PasswordField('New Password (Leave blank to keep unchanged)', validators=[Optional(), Length(min=8)])
    user_id = HiddenField('User ID')
    submit = SubmitField('Update User')


class SystemSettingsForm(FlaskForm):
    """Form for system settings."""
    # General Settings
    site_name = StringField('Site Name', validators=[DataRequired(), Length(max=100)])
    site_description = TextAreaField('Site Description', validators=[Length(max=500)])
    max_users = IntegerField('Maximum Users', validators=[DataRequired(), NumberRange(min=1)])
    session_timeout = IntegerField('Session Timeout (minutes)', validators=[DataRequired(), NumberRange(min=5)])
    
    # ML Model Settings
    model_confidence_threshold = FloatField('Model Confidence Threshold', validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    model_version = StringField('Model Version', validators=[DataRequired(), Length(max=50)])
    enable_auto_retrain = BooleanField('Enable Auto Retrain')
    
    # Email Settings
    smtp_server = StringField('SMTP Server', validators=[Length(max=100)])
    smtp_port = IntegerField('SMTP Port', validators=[NumberRange(min=1, max=65535)])
    smtp_username = StringField('SMTP Username', validators=[Length(max=100)])
    smtp_password = PasswordField('SMTP Password', validators=[Length(max=100)])
    enable_email_notifications = BooleanField('Enable Email Notifications')
    
    # Security Settings
    password_min_length = IntegerField('Minimum Password Length', validators=[DataRequired(), NumberRange(min=6)])
    max_login_attempts = IntegerField('Maximum Login Attempts', validators=[DataRequired(), NumberRange(min=1)])
    enable_two_factor = BooleanField('Enable Two-Factor Authentication')
    
    submit = SubmitField('Save Settings')


# Routes
@app.route('/')
def index():
    """Home page."""
    if current_user.is_authenticated and current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    
    form = CBCForm()
    return render_template('index.html', form=form)


@app.route('/about')
def about():
    """About page with chart data."""
    # Temporary static dataset
    hemoglobin_values = [13.2, 12.8, 13.5, 14.0, 13.7]
    dates = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]

    return render_template(
        'about.html',
        hemoglobin_values=hemoglobin_values,
        dates=dates
    )



@app.route('/faq')
def faq():
    """FAQ page."""
    return render_template('faq.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        success, result = db.verify_user(form.username.data, form.password.data)
        if success:
            user = User(result)
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                # Redirect to admin dashboard if user is admin, otherwise dashboard
                if user.is_admin:
                    next_page = url_for('admin_dashboard')
                else:
                    next_page = url_for('dashboard')
            return redirect(next_page)
        else:
            flash(result)
    
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        success, result = db.create_user(
            username=form.username.data,
            password=form.password.data,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            gender=form.gender.data,
            date_of_birth=form.date_of_birth.data,
            medical_id=form.medical_id.data
        )
        
        if success:
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        else:
            flash(f'Registration failed: {result}')
    
    return render_template('register.html', form=form)


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard."""
    # Get user's recent classification history
    history = db.get_user_classification_history(current_user.id, limit=5)
    
    # Get medical data
    medical_data = db.get_medical_data(current_user.id)
    
    # Create form for hemoglobin input
    form = CBCForm()
    
    # Prepare data for charts
    hemoglobin_values = []
    rbc_values = []
    hct_values = []
    mcv_values = []
    dates = []
    
    for record in history:
        hemoglobin_values.append(record['hgb'])
        rbc_values.append(record['rbc'])
        mcv_values.append(record['mcv'])
        hct_values.append(record['hct'])
        # Convert SQLite timestamp string to datetime
        created_at = datetime.strptime(record['created_at'], '%Y-%m-%d %H:%M:%S')
        dates.append(created_at.strftime('%Y-%m-%d'))
    
    # Reverse lists to show chronological order
    hemoglobin_values.reverse()
    dates.reverse()
    return render_template(
        'dashboard.html',
        history=history,
        medical_data=medical_data,
        hemoglobin_values=hemoglobin_values,
        rbc_values=rbc_values,
        mcv_values=mcv_values,
        hct_values=hct_values,
        dates=dates,
        form=form
    )


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page."""
    form = ProfileForm()
    
    if request.method == 'GET':
        # Pre-populate form with current user data
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email
        form.gender.data = current_user.gender
        form.date_of_birth.data = current_user.date_of_birth
        form.medical_id.data = current_user.medical_id
    
    if form.validate_on_submit():
        # Check current password if provided
        if form.current_password.data:
            success, _ = db.verify_user(current_user.username, form.current_password.data)
            if not success:
                flash('Current password is incorrect.')
                return redirect(url_for('profile'))
            
            # If new password is provided, update it
            if form.new_password.data:
                success, result = db.update_user(
                    current_user.id,
                    password=form.new_password.data
                )
                if not success:
                    flash(f'Password update failed: {result}')
                    return redirect(url_for('profile'))
        
        # Update user information
        success, result = db.update_user(
            current_user.id,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            gender=form.gender.data,
            date_of_birth=form.date_of_birth.data,
            medical_id=form.medical_id.data
        )
        
        if success:
            flash('Profile updated successfully!')
            return redirect(url_for('profile'))
        else:
            flash(f'Profile update failed: {result}')
    
    return render_template('profile.html', form=form)


@app.route('/medical-data', methods=['GET', 'POST'])
@login_required
def medical_data():
    """User medical data page."""
    form = MedicalDataForm()
    medical_data = db.get_medical_data(current_user.id)
    
    if request.method == 'GET' and medical_data:
        # Pre-populate form with current medical data
        form.height.data = medical_data.get('height')
        form.weight.data = medical_data.get('weight')
        form.blood_type.data = medical_data.get('blood_type')
        form.medical_conditions.data = medical_data.get('medical_conditions')
        form.medications.data = medical_data.get('medications')
    
    if form.validate_on_submit():
        # Update medical data
        success, result = db.update_medical_data(
            current_user.id,
            height=form.height.data,
            weight=form.weight.data,
            blood_type=form.blood_type.data,
            medical_conditions=form.medical_conditions.data,
            medications=form.medications.data
        )
        
        if success:
            flash('Medical data updated successfully!')
            return redirect(url_for('medical_data'))
        else:
            flash(f'Medical data update failed: {result}')
    
    return render_template('medical_data.html', form=form, medical_data=medical_data)


@app.route('/history')
@login_required
def history():
    """User classification history page."""
    # Get user's classification history
    records = db.get_user_classification_history(current_user.id, limit=50)
    
    return render_template('history.html', records=records)


@app.route('/classify', methods=['POST'])
@login_required
def classify():
    """Process the hemoglobin data and make a prediction."""
    form = CBCForm()
    
    if form.validate_on_submit():
        hemoglobin = form.hemoglobin.data
        notes = form.notes.data
        
        # Get prediction from model
        result = anemia_model.predict(hemoglobin)
        
        # Save the record to database
        record_id = db.add_classification_record(
            user_id=current_user.id,
            hemoglobin=hemoglobin,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            recommendation=result['recommendation'],
            notes=notes
        )
        
        # Emit real-time update via WebSocket
        classification_data = {
            'id': record_id,
            'hemoglobin': hemoglobin,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'recommendation': result['recommendation'],
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'notes': notes,
            'user_id': current_user.id,
            'username': current_user.username
        }
        
        # Emit to the user's room
        socketio.emit('new_classification', classification_data, room=str(current_user.id))
        
        # Also emit to admin room if this is a normal user
        if not current_user.is_admin:
            socketio.emit('admin_new_classification', classification_data, room='admin_room')
        
        # Redirect to result page
        return redirect(url_for('result', record_id=record_id))
    
    return redirect(url_for('index'))

@app.route('/rfcclasify', methods=['POST'])
@login_required
def rfcclasify():
    """Process the hemoglobin data and make a prediction."""
    #try:
    form = CBCForm()
    
    submit = request.form.get('submit')
    
    if submit is None:
        return "Error: Submit button not clicked or missing in the form data.", 400

    wbc = request.form.get("wbc")
    rbc = request.form.get("rbc")
    hgb = request.form.get("hgb")
    hct = request.form.get("hct")
    mcv = request.form.get("mcv")
    mch = request.form.get("mch")
    mchc = request.form.get("mchc")
    plt = request.form.get("plt")
    notes = request.form.get("notes")  # Corrected from "mcv" to "notes"

    # # Example birth date as a string
    # birth_date_str = current_user.date_of_birth

    # # Convert string to date object
    # birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()

    # # Get today's date
    # today = datetime.today().date()

    # # Calculate age
    # age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    # gender = 0 if current_user.gender.lower() == "female" else 0
    
    model = joblib.load('best_rf_anemia_model.joblib')

    input_data = pd.DataFrame([{
        'WBC': float(wbc),
        'RBC': float(rbc),
        'HGB': float(hgb),
        'HCT': float(hct),
        'MCV': float(mcv),
        'MCH': float(mch),
        'MCHC': float(mchc),
        'PLT': float(plt)
    }])

    # Predict class and probability
    probabilities = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]

    # Label and confidence
    label_mapping = {0: 'Anemia', 1: 'Normal'}
    predicted_label = label_mapping[prediction]
    confidence = round(probabilities[prediction] * 100, 2)

    # Recommendations dictionary
    recommendations = {
        'Normal': "Maintain a healthy diet rich in iron, vitamin B12, and folate.",
        'Mild': "Consider dietary adjustments to increase iron intake and monitor hemoglobin "
                "levels in 1-2 months. Foods rich in iron include red meat, spinach, and legumes.",
        'Moderate': "Medical consultation recommended. Iron supplements may be prescribed. "
                    "Further testing might be needed to determine the underlying cause.",
        'Severe': "Emergency medical care required. Immediate consultation with a healthcare "
                "provider is necessary as severe anemia can lead to serious complications."
    }

    # Determine severity if Anemia
    if predicted_label == 'Normal':
        final_recommendation = recommendations['Normal']
        severity = 'None'
    else:
        if confidence >= 90:
            severity = 'Severe'
        elif confidence >= 75:
            severity = 'Moderate'
        else:
            severity = 'Mild'
        predicted_label = severity
        final_recommendation = recommendations[severity]

    # Output
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence}%")
    print(f"Recommendation: {final_recommendation}")


        # Save the record to database
    record_id = db.add_classification_record(
        user_id=current_user.id,
        wbc=float(wbc),
        rbc=float(rbc),
        hgb=float(hgb),
        hct=float(hct),
        mcv=float(mcv),
        mch=float(mch),
        mchc=float(mchc),
        plt=float(plt),
        predicted_class=predicted_label,
        confidence=confidence/100,
        recommendation=final_recommendation,
        notes=notes
    )

    
    

        # Emit real-time update via WebSocket
    classification_data = {
        'id': record_id,
        'user_id': current_user.id,
        'username': current_user.username,
        'wbc': wbc,
        'rbc': rbc,
        'hgb': hgb,
        'hct': hct,
        'mcv': mcv,
        'mch': mch,
        'mchc': mchc,
        'plt': plt,
        'predicted_class': predicted_label,
        'confidence': confidence / 100,
        'recommendation': final_recommendation,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'notes': notes
    }

    # Emit to the user's room
    socketio.emit('new_classification', classification_data, room=str(current_user.id))
    
    # Also emit to admin room if this is a normal user
    if not current_user.is_admin:
        socketio.emit('admin_new_classification', classification_data, room='admin_room')
    
    # Redirect to result page
    return redirect(url_for('result', record_id=record_id))


@app.route('/xgbclasify', methods=['POST'])
@login_required
def xgbclasify():
    """Process the hemoglobin data and make a prediction."""
    #try:
    form = CBCForm()
    
    submit = request.form.get('submit')
    
    if submit is None:
        return "Error: Submit button not clicked or missing in the form data.", 400
    
    wbc = float(request.form.get("WBC"))
    rbc = float(request.form.get("RBC"))
    hgb = float(request.form.get("HEMOGLOBIN"))
    hct = float(request.form.get("HEMATOCRIT"))
    mcv = float(request.form.get("MCV"))
    mch = float(request.form.get("MCH"))
    mchc = float(request.form.get("MCHC"))
    plt = float(request.form.get("PLATELET"))
    neutrophils = float(request.form.get("NEUTROPHILS"))
    lymphocytes = float(request.form.get("LYMPHOCYTES"))
    monocytes = float(request.form.get("MONOCYTES"))
    eosinophils = float(request.form.get("EUSONIPHILS"))
    basophil = float(request.form.get("BASOPHIL"))
    immature_granulocytes = float(request.form.get("IMMATURE_GRANULYTES"))
    notes = request.form.get("notes")  # Corrected from "mcv" to "notes"

    # Example birth date as a string
    birth_date_str = current_user.date_of_birth

    # Convert string to date object
    birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()

    # Get today's date
    today = datetime.today().date()

    # Calculate age
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    gender = 1 if current_user.gender.lower() == "female" else 0

    user_input = [
        age,
        gender,
        wbc,
        rbc,
        hgb,
        hct,
        mcv,
        mch,
        mchc,
        plt,
        neutrophils,
        lymphocytes,
        monocytes,
        eosinophils,
        basophil,
        immature_granulocytes
    ]
    print(user_input)
    predicted_label,confidence_scores = xgboost_predict(user_input)
    # Recommendations dictionary
    recommendations = {
        'Normal': "Maintain a healthy diet rich in iron, vitamin B12, and folate.",
        'Mild': "Consider dietary adjustments to increase iron intake and monitor hemoglobin "
                "levels in 1-2 months. Foods rich in iron include red meat, spinach, and legumes.",
        'Moderate': "Medical consultation recommended. Iron supplements may be prescribed. "
                    "Further testing might be needed to determine the underlying cause.",
        'Severe': "Emergency medical care required. Immediate consultation with a healthcare "
                "provider is necessary as severe anemia can lead to serious complications."
    }

    final_recommendation = recommendations[predicted_label]

    # Output
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {round(confidence_scores*100, 2)}%")
    print(f"Recommendation: {final_recommendation}")


        # Save the record to database
    record_id = db.add_classification_record(
        user_id=current_user.id,
        wbc = float(wbc),
        rbc = float(rbc),
        hgb = float(hgb),
        hct = float(hct),
        mcv = float(mcv),
        mch = float(mch),
        mchc = float(mchc),
        plt = float(plt),
        neutrophils = float(neutrophils),
        lymphocytes = float(lymphocytes),
        monocytes = float(monocytes),
        eosinophils = float(eosinophils),
        basophil = float(basophil),
        immature_granulocytes = float(immature_granulocytes),
        predicted_class=predicted_label,
        confidence=float(confidence_scores),
        recommendation=final_recommendation,
        notes=notes
    )

    
    

        # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        return obj
    
    # Emit real-time update via WebSocket
    classification_data = {
        'id': record_id,
        'user_id': current_user.id,
        'username': current_user.username,
        'wbc': convert_numpy_types(wbc),
        'rbc': convert_numpy_types(rbc),
        'hgb': convert_numpy_types(hgb),
        'hct': convert_numpy_types(hct),
        'mcv': convert_numpy_types(mcv),
        'mch': convert_numpy_types(mch),
        'mchc': convert_numpy_types(mchc),
        'plt': convert_numpy_types(plt),
        'neutrophils': convert_numpy_types(neutrophils),
        'lymphocytes': convert_numpy_types(lymphocytes),
        'monocytes': convert_numpy_types(monocytes),
        'eosinophils': convert_numpy_types(eosinophils),
        'basophil': convert_numpy_types(basophil),
        'immature_granulocytes': convert_numpy_types(immature_granulocytes),
        'predicted_class': predicted_label,
        'confidence': convert_numpy_types(confidence_scores),
        'recommendation': final_recommendation,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'notes': notes,
        'age': age,
        'gender': current_user.gender
    }

    # Emit to the user's room
    socketio.emit('new_classification', classification_data, room=str(current_user.id))
    
    # Also emit to admin room if this is a normal user
    if not current_user.is_admin:
        socketio.emit('admin_new_classification', classification_data, room='admin_room')
    
    # Automatically send email with results
    try:
        user_data = db.get_user_by_id(current_user.id)
        if user_data:
            # Prepare record data for email
            record_data = {
                'predicted_class': predicted_label,
                'confidence': float(confidence_scores),
                'wbc': float(wbc),
                'rbc': float(rbc),
                'hgb': float(hgb),
                'hct': float(hct),
                'mcv': float(mcv),
                'mch': float(mch),
                'mchc': float(mchc),
                'plt': float(plt),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'notes': notes
            }
            
            # Send email
            success, message = send_result_email_helper(
                record_id, 
                user_data['email'], 
                f"{user_data['first_name']} {user_data['last_name']}".strip() or user_data['username'],
                record_data
            )
            
            if success:
                logger.info(f"Auto-email sent successfully to {user_data['email']}")
            else:
                logger.warning(f"Auto-email failed: {message}")
    except Exception as e:
        logger.error(f"Error sending auto-email: {str(e)}")
    
    # Redirect to result page
    return redirect(url_for('result', record_id=record_id))            

@app.route('/api/classification-stats')
@login_required
def get_classification_stats():
    """Get gender and age statistics for visualization."""
    try:
        # Get all classification records
        records = db.get_all_classification_history(limit=1000)
        
        # Process data for visualization
        gender_stats = {'Male': 0, 'Female': 0, 'Other': 0}
        age_groups = {'0-18': 0, '19-30': 0, '31-50': 0, '51-70': 0, '70+': 0}
        classification_stats = {'Normal': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0}
        
        for record in records:
            # Get user data
            user_data = db.get_user_by_id(record['user_id'])
            if user_data:
                # Gender statistics (normalize to title-case buckets)
                raw_gender = (user_data.get('gender') or 'other').strip().lower()
                if raw_gender in ('male', 'm'):
                    bucket = 'Male'
                elif raw_gender in ('female', 'f'):
                    bucket = 'Female'
                else:
                    bucket = 'Other'
                gender_stats[bucket] += 1
                
                # Age calculation (if date_of_birth is available)
                if user_data.get('date_of_birth'):
                    try:
                        birth_date = datetime.strptime(user_data['date_of_birth'], "%Y-%m-%d").date()
                        today = datetime.today().date()
                        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                        
                        # Age groups
                        if age <= 18:
                            age_groups['0-18'] += 1
                        elif age <= 30:
                            age_groups['19-30'] += 1
                        elif age <= 50:
                            age_groups['31-50'] += 1
                        elif age <= 70:
                            age_groups['51-70'] += 1
                        else:
                            age_groups['70+'] += 1
                    except:
                        pass
            
            # Classification statistics
            predicted_class = record.get('predicted_class', 'Normal')
            if predicted_class in classification_stats:
                classification_stats[predicted_class] += 1
        
        return jsonify({
            'success': True,
            'data': {
                'gender_stats': gender_stats,
                'age_groups': age_groups,
                'classification_stats': classification_stats
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting classification stats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/xgbclasifytry', methods=['POST'])

def xgb_try_clasify():
    """Process the hemoglobin data and make a prediction."""
    #try:
    form = CBCForm()
    
    submit = request.form.get('submit')
    
    if submit is None:
        return "Error: Submit button not clicked or missing in the form data.", 400
    age = int(request.form.get("age"))
    wbc = float(request.form.get("WBC"))
    rbc = float(request.form.get("RBC"))
    hgb = float(request.form.get("HEMOGLOBIN"))
    hct = float(request.form.get("HEMATOCRIT"))
    mcv = float(request.form.get("MCV"))
    mch = float(request.form.get("MCH"))
    mchc = float(request.form.get("MCHC"))
    plt = float(request.form.get("PLATELET"))
    neutrophils = float(request.form.get("NEUTROPHILS"))
    lymphocytes = float(request.form.get("LYMPHOCYTES"))
    monocytes = float(request.form.get("MONOCYTES"))
    eosinophils = float(request.form.get("EUSONIPHILS"))
    basophil = float(request.form.get("BASOPHIL"))
    immature_granulocytes = float(request.form.get("IMMATURE_GRANULYTES") or 0.0)
    #notes = request.form.get("notes")  # Corrected from "mcv" to "notes"

    
    gender = 1 if request.form.get("gender").lower() == "female" else 0

    user_input = [
        age,
        gender,
        wbc,
        rbc,
        hgb,
        hct,
        mcv,
        mch,
        mchc,
        plt,
        neutrophils,
        lymphocytes,
        monocytes,
        eosinophils,
        basophil,
        immature_granulocytes
    ]
    print(user_input)
    predicted_label,confidence_scores = xgboost_predict(user_input)
    # Recommendations dictionary
    recommendations = {
        'Normal': "Maintain a healthy diet rich in iron, vitamin B12, and folate.",
        'Mild': "Consider dietary adjustments to increase iron intake and monitor hemoglobin "
                "levels in 1-2 months. Foods rich in iron include red meat, spinach, and legumes.",
        'Moderate': "Medical consultation recommended. Iron supplements may be prescribed. "
                    "Further testing might be needed to determine the underlying cause.",
        'Severe': "Emergency medical care required. Immediate consultation with a healthcare "
                "provider is necessary as severe anemia can lead to serious complications."
    }

    

    final_recommendation = recommendations[predicted_label]

    

    # Output
    
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {round(confidence_scores*100, 2)}%")
    print(f"Recommendation: {final_recommendation}")

    cbc_results_summary = {
        "Normal": "Normal: Your CBC results, including your hemoglobin, hematocrit, red blood cell count, and other related values, are all within the normal range. This means your blood is healthy and able to carry oxygen properly throughout your body.",
        
        "Mild Anemia": "Mild Anemia: Some of your blood test results, such as your hemoglobin or red blood cell count, are just slightly below normal. While you might not feel many symptoms yet, these early changes suggest your blood isn't carrying oxygen quite as efficiently, so we'll monitor it and take steps if needed.",
        
        "Moderate Anemia": "Moderate Anemia: Several parts of your CBC, including hemoglobin, hematocrit, and red cell indices like MCV or MCH, show moderate changes. These results explain why you might be feeling more tired, weak, or short of breath, and we'll need to start treatment to correct it.",
        
        "Severe Anemia": "Severe Anemia: Your CBC shows multiple markers—like hemoglobin, red blood cell count, and hematocrit—are well below normal. This means your body isn't getting enough oxygen, which can cause serious symptoms, so we need to act quickly to manage and treat the cause."
    }
    if predicted_label != "Normal": 
        predicted_label += " Anemia"


    record_id = 0

        # Emit real-time update via WebSocket
    record = {
        'id': record_id,
        'wbc': wbc,
        'rbc': rbc,
        'hgb': hgb,
        'hct': hct,
        'mcv': mcv,
        'mch': mch,
        'mchc': mchc,
        'plt': plt,
        'neutrophils': neutrophils,
        'lymphocytes': lymphocytes,
        'monocytes': monocytes,
        'eosinophils': eosinophils,
        'basophil': basophil,
        'immature_granulocytes': immature_granulocytes,
        'predicted_class': predicted_label,
        'confidence': confidence_scores,
        'recommendation': final_recommendation,
        'definition': cbc_results_summary[predicted_label],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    
    return render_template(
        'result_trial.html',
        record=record,
    )
    



@app.route('/result/<int:record_id>')
@login_required
def result(record_id):
    """Display classification result."""
    # Get the classification record
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM classification_history WHERE id = ? AND user_id = ?",
        (record_id, current_user.id)
    )
    
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        flash('Record not found.')
        return redirect(url_for('dashboard'))
    
    # Convert to dict
    record = dict(record)
    print(record)
    # Generate visualization if enabled
    visualization = None
    if db.get_system_setting('visualization_enabled') == 'true':
        # Get tree visualization
        visualization = anemia_model.get_tree_visualization()
    

    cbc_results_summary = {
        "Normal": "Normal: Your CBC results, including your hemoglobin, hematocrit, red blood cell count, and other related values, are all within the normal range. This means your blood is healthy and able to carry oxygen properly throughout your body.",
        
        "Mild Anemia": "Mild Anemia: Some of your blood test results, such as your hemoglobin or red blood cell count, are just slightly below normal. While you might not feel many symptoms yet, these early changes suggest your blood isn't carrying oxygen quite as efficiently, so we'll monitor it and take steps if needed.",
        
        "Moderate Anemia": "Moderate Anemia: Several parts of your CBC, including hemoglobin, hematocrit, and red cell indices like MCV or MCH, show moderate changes. These results explain why you might be feeling more tired, weak, or short of breath, and we'll need to start treatment to correct it.",
        
        "Severe Anemia": "Severe Anemia: Your CBC shows multiple markers—like hemoglobin, red blood cell count, and hematocrit—are well below normal. This means your body isn't getting enough oxygen, which can cause serious symptoms, so we need to act quickly to manage and treat the cause."
    }
    predicted_label = record["predicted_class"]
    if predicted_label != "Normal": 
        predicted_label += " Anemia"

    record["predicted_class"] = predicted_label

    record["definition"] = cbc_results_summary[predicted_label]
    
    return render_template(
        'result.html',
        record=record,
        visualization=visualization
    )


@app.route('/api/classify', methods=['POST'])
@login_required
def api_classify():
    """API endpoint for anemia classification."""
    try:
        data = request.get_json()
        if not data or 'hemoglobin' not in data:
            return jsonify({'error': 'Missing hemoglobin value'}), 400
        
        hemoglobin = float(data['hemoglobin'])
        notes = data.get('notes', '')
        
        # Validate hemoglobin range
        if hemoglobin < 1 or hemoglobin > 25:
            return jsonify({'error': 'Hemoglobin value out of valid range (1-25 g/dL)'}), 400
        
        # Get prediction from model
        result = anemia_model.predict(hemoglobin)
        
        # Save the record to database
        record_id = db.add_classification_record(
            user_id=current_user.id,
            hemoglobin=hemoglobin,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            recommendation=result['recommendation'],
            notes=notes
        )
        
        # Add record_id to result
        result['record_id'] = record_id
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API classification error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Admin routes
@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get page parameter for pagination
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    
    # Get system statistics
    stats = db.get_statistics()
    
    # Get recent classifications with pagination
    recent_data = db.get_recent_classifications(page=page, per_page=5)
    
    # Get chart data
    charts_data = db.get_admin_dashboard_charts()
    
    return render_template('admin/dashboard.html', stats=stats, recent_data=recent_data, charts_data=charts_data)


@app.route('/admin/users')
@login_required
def admin_users():
    """Admin user management page."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    users_data = db.get_users_paginated(page=page, per_page=5)
    
    return render_template('admin/users.html', users_data=users_data)


@app.route('/admin/api/username-exists')
@login_required
def admin_api_username_exists():
    """Admin API: check if a username exists, optionally excluding a user id."""
    if not current_user.is_admin:
        return jsonify({ 'success': False, 'error': 'Access denied' }), 403
    username = (request.args.get('username') or '').strip()
    exclude_id = (request.args.get('exclude_id') or '').strip()
    if not username:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_username(username)
    if not user:
        return jsonify({ 'success': True, 'exists': False })
    if exclude_id and str(user.get('id')) == str(exclude_id):
        return jsonify({ 'success': True, 'exists': False })
    return jsonify({ 'success': True, 'exists': True })


@app.route('/admin/api/email-exists')
@login_required
def admin_api_email_exists():
    """Admin API: check if an email exists, optionally excluding a user id."""
    if not current_user.is_admin:
        return jsonify({ 'success': False, 'error': 'Access denied' }), 403
    email = (request.args.get('email') or '').strip()
    exclude_id = (request.args.get('exclude_id') or '').strip()
    if not email:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_email(email)
    if not user:
        return jsonify({ 'success': True, 'exists': False })
    if exclude_id and str(user.get('id')) == str(exclude_id):
        return jsonify({ 'success': True, 'exists': False })
    return jsonify({ 'success': True, 'exists': True })

@app.route('/admin/api/medical-id-exists')
@login_required
def admin_api_medical_id_exists():
    """Admin API: check if a medical ID exists, optionally excluding a user id."""
    if not current_user.is_admin:
        return jsonify({ 'success': False, 'error': 'Access denied' }), 403
    medical_id = (request.args.get('medical_id') or '').strip()
    exclude_id = (request.args.get('exclude_id') or '').strip()
    if not medical_id:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_medical_id(medical_id)
    if not user:
        return jsonify({ 'success': True, 'exists': False })
    if exclude_id and str(user.get('id')) == str(exclude_id):
        return jsonify({ 'success': True, 'exists': False })
    return jsonify({ 'success': True, 'exists': True })


@app.route('/admin/user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def admin_edit_user(user_id):
    """Admin edit user page."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get user data
    user_data = db.get_user(user_id)
    if not user_data:
        flash('User not found.')
        return redirect(url_for('admin_users'))
    
    form = AdminUserForm()
    
    if request.method == 'GET':
        # Pre-populate form with user data
        form.username.data = user_data['username']
        form.email.data = user_data['email']
        form.first_name.data = user_data['first_name']
        form.last_name.data = user_data['last_name']
        form.gender.data = user_data['gender']
        form.date_of_birth.data = user_data['date_of_birth']
        form.medical_id.data = user_data['medical_id']
        form.is_admin.data = bool(user_data['is_admin'])
        form.user_id.data = user_id
    
    if form.validate_on_submit():
        # Update user
        update_data = {
            'username': form.username.data,
            'email': form.email.data,
            'first_name': form.first_name.data,
            'last_name': form.last_name.data,
            'gender': form.gender.data,
            'date_of_birth': form.date_of_birth.data,
            'medical_id': form.medical_id.data,
            'is_admin': 1 if form.is_admin.data else 0
        }
        
        # Add password if provided
        if form.password.data:
            update_data['password'] = form.password.data
        
        success, result = db.update_user(user_id, **update_data)
        
        if success:
            flash('User updated successfully!')
            return redirect(url_for('admin_users'))
        else:
            flash(f'User update failed: {result}')
    
    return render_template('admin/edit_user.html', form=form, user=user_data)


@app.route('/admin/user/<int:user_id>/details')
@login_required
def admin_user_details(user_id):
    """Get detailed user information for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get user data
    user_data = db.get_user(user_id)
    if not user_data:
        return jsonify({'success': False, 'error': 'User not found'}), 404
    
    # Get user's medical data
    medical_data = db.get_medical_data(user_id)
    
    # Get user's classification history (last 10)
    classification_history = db.get_user_classification_history(user_id, limit=10)
    
    # Get user's chat conversations
    conversations = simple_chat.get_user_conversations(user_id, is_admin=False)
    
    # Prepare response data
    user_details = {
        'user': {
            'id': user_data['id'],
            'username': user_data['username'],
            'email': user_data['email'],
            'first_name': user_data['first_name'],
            'last_name': user_data['last_name'],
            'gender': user_data['gender'],
            'date_of_birth': user_data['date_of_birth'],
            'medical_id': user_data['medical_id'],
            'is_admin': user_data['is_admin'],
            'created_at': user_data['created_at'],
            'last_login': user_data['last_login']
        },
        'medical_data': medical_data,
        'classification_history': classification_history,
        'conversations': conversations
    }
    
    return jsonify({'success': True, 'data': user_details})


@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    """Admin delete user."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Prevent self-deletion
    if user_id == current_user.id:
        return jsonify({'success': False, 'error': 'Cannot delete your own account'}), 400
    
    # Delete the user
    success, message = db.delete_user(user_id)
    
    if success:
        logger.info(f"Admin {current_user.username} deleted user ID {user_id}")
        return jsonify({'success': True, 'message': message})
    else:
        logger.error(f"Failed to delete user ID {user_id}: {message}")
        return jsonify({'success': False, 'error': message}), 400


@app.route('/admin/history')
@login_required
def admin_history():
    """Admin classification history page."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get page parameter for pagination
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    
    # Get paginated classification history
    history_data = db.get_classification_history_paginated(page=page, per_page=5)
    
    # Get system statistics (same as dashboard)
    stats = db.get_statistics()
    
    # Calculate additional statistics for the history page
    total_records = history_data['total']
    anemic_cases = stats['anemic_cases']
    normal_cases = stats['normal_cases']
    
    # Calculate anemia rate
    if total_records > 0:
        anemia_rate = (anemic_cases / total_records) * 100
    else:
        anemia_rate = 0.0
    
    return render_template('admin/history.html', 
                         history_data=history_data,
                         total_records=total_records,
                         anemic_cases=anemic_cases,
                         normal_cases=normal_cases,
                         anemia_rate=anemia_rate,
                         stats=stats)


@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
def admin_settings():
    """Admin system settings page."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    form = SystemSettingsForm()
    
    if request.method == 'GET':
        # Pre-populate form with current settings
        form.site_name.data = db.get_system_setting('site_name') or 'AnemoCheck'
        form.site_description.data = db.get_system_setting('site_description') or 'Anemia Detection System'
        form.max_users.data = int(db.get_system_setting('max_users') or 1000)
        form.session_timeout.data = int(db.get_system_setting('session_timeout') or 30)
        form.model_confidence_threshold.data = float(db.get_system_setting('model_confidence_threshold') or 0.8)
        form.model_version.data = db.get_system_setting('model_version') or '1.0.0'
        form.enable_auto_retrain.data = db.get_system_setting('enable_auto_retrain') == 'true'
        form.smtp_server.data = db.get_system_setting('smtp_server') or ''
        form.smtp_port.data = int(db.get_system_setting('smtp_port') or 587)
        form.smtp_username.data = db.get_system_setting('smtp_username') or ''
        form.enable_email_notifications.data = db.get_system_setting('enable_email_notifications') == 'true'
        form.password_min_length.data = int(db.get_system_setting('password_min_length') or 8)
        form.max_login_attempts.data = int(db.get_system_setting('max_login_attempts') or 5)
        form.enable_two_factor.data = db.get_system_setting('enable_two_factor') == 'true'
    
    if form.validate_on_submit():
        logger.info("Admin settings form submitted")
        logger.info(f"SMTP Server: {form.smtp_server.data}")
        logger.info(f"SMTP Port: {form.smtp_port.data}")
        logger.info(f"SMTP Username: {form.smtp_username.data}")
        logger.info(f"Enable Email Notifications: {form.enable_email_notifications.data}")
        
        # Update settings
        db.update_system_setting('site_name', form.site_name.data, current_user.id)
        db.update_system_setting('site_description', form.site_description.data, current_user.id)
        db.update_system_setting('max_users', str(form.max_users.data), current_user.id)
        db.update_system_setting('session_timeout', str(form.session_timeout.data), current_user.id)
        db.update_system_setting('model_confidence_threshold', str(form.model_confidence_threshold.data), current_user.id)
        db.update_system_setting('model_version', form.model_version.data, current_user.id)
        db.update_system_setting('enable_auto_retrain', 'true' if form.enable_auto_retrain.data else 'false', current_user.id)
        
        # Email settings
        db.update_system_setting('smtp_server', form.smtp_server.data, current_user.id)
        db.update_system_setting('smtp_port', str(form.smtp_port.data), current_user.id)
        db.update_system_setting('smtp_username', form.smtp_username.data, current_user.id)
        if form.smtp_password.data:
            db.update_system_setting('smtp_password', form.smtp_password.data, current_user.id)
        db.update_system_setting('enable_email_notifications', 'true' if form.enable_email_notifications.data else 'false', current_user.id)
        
        db.update_system_setting('password_min_length', str(form.password_min_length.data), current_user.id)
        db.update_system_setting('max_login_attempts', str(form.max_login_attempts.data), current_user.id)
        db.update_system_setting('enable_two_factor', 'true' if form.enable_two_factor.data else 'false', current_user.id)
        
        # Verify settings were saved
        saved_smtp_server = db.get_system_setting('smtp_server')
        saved_enable_notifications = db.get_system_setting('enable_email_notifications')
        logger.info(f"Saved SMTP Server: {saved_smtp_server}")
        logger.info(f"Saved Enable Notifications: {saved_enable_notifications}")
        
        flash('Settings updated successfully!')
        return redirect(url_for('admin_settings'))
    
    return render_template('admin/settings.html', form=form)


# Email sending functionality
def send_result_email_helper(record_id, user_email, user_name, record_data):
    """Send anemia test result email to user."""
    try:
        # Get email settings from database
        smtp_server = db.get_system_setting('smtp_server')
        smtp_port = int(db.get_system_setting('smtp_port') or 587)
        smtp_username = db.get_system_setting('smtp_username')
        smtp_password = db.get_system_setting('smtp_password')
        enable_notifications = db.get_system_setting('enable_email_notifications') == 'true'
        
        if not enable_notifications or not smtp_server or not smtp_username:
            return False, "Email notifications are not configured or enabled"
        
        # Create email content
        subject = f"AnemoCheck - Your Anemia Test Result ({record_data['predicted_class']})"
        
        # Create HTML email content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Anemia Test Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #c62828; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .result-box {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .classification {{ font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 5px; }}
                .normal {{ background-color: #d4edda; color: #155724; }}
                .mild {{ background-color: #fff3cd; color: #856404; }}
                .moderate {{ background-color: #f8d7da; color: #721c24; }}
                .severe {{ background-color: #f5c6cb; color: #721c24; }}
                .values-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .values-table th, .values-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .values-table th {{ background-color: #f2f2f2; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AnemoCheck - Anemia Test Result</h1>
                </div>
                <div class="content">
                    <h2>Hello {user_name},</h2>
                    <p>Your anemia classification test has been completed. Here are your results:</p>
                    
                    <div class="result-box">
                        <h3>Classification Result</h3>
                        <div class="classification {'normal' if record_data['predicted_class'] == 'Normal' else 'mild' if record_data['predicted_class'] == 'Mild' else 'moderate' if record_data['predicted_class'] == 'Moderate' else 'severe' if record_data['predicted_class'] == 'Severe' else ''}">
                            {record_data['predicted_class']}
                        </div>
                        <p style="text-align: center; margin-top: 10px;">
                            <strong>Confidence: {record_data['confidence']:.1%}</strong>
                        </p>
                    </div>
                    
                    <div class="result-box">
                        <h3>Complete Blood Count (CBC) Values</h3>
                        <table class="values-table">
                            <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
                            <tr><td>White Blood Cell Count (WBC)</td><td>{record_data['wbc']:.2f}</td><td>10³/µL</td></tr>
                            <tr><td>Red Blood Cell Count (RBC)</td><td>{record_data['rbc']:.2f}</td><td>million/µL</td></tr>
                            <tr><td>Hemoglobin (HGB)</td><td>{record_data['hgb']:.2f}</td><td>g/dL</td></tr>
                            <tr><td>Hematocrit (HCT)</td><td>{record_data['hct']:.2f}</td><td>%</td></tr>
                            <tr><td>Mean Corpuscular Volume (MCV)</td><td>{record_data['mcv']:.2f}</td><td>fL</td></tr>
                            <tr><td>Mean Corpuscular Hemoglobin (MCH)</td><td>{record_data['mch']:.2f}</td><td>pg</td></tr>
                            <tr><td>Mean Corpuscular Hemoglobin Concentration (MCHC)</td><td>{record_data['mchc']:.2f}</td><td>g/dL</td></tr>
                            <tr><td>Platelet Count (PLT)</td><td>{record_data['plt']:.2f}</td><td>10³/µL</td></tr>
                        </table>
                    </div>
                    
                    {f'<div class="result-box"><h3>Notes</h3><p>{record_data.get("notes", "")}</p></div>' if record_data.get("notes") else ''}
                    
                    <div class="result-box">
                        <h3>Important Information</h3>
                        <p><strong>Please note:</strong> This is an AI-powered screening tool and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.</p>
                        <p><strong>Test Date:</strong> {record_data['created_at']}</p>
                    </div>
                </div>
                <div class="footer">
                    <p>This email was sent from AnemoCheck - Anemia Detection System</p>
                    <p>For support, please contact your healthcare provider</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        text_content = f"""
        AnemoCheck - Anemia Test Result
        
        Hello {user_name},
        
        Your anemia classification test has been completed. Here are your results:
        
        Classification Result: {record_data['predicted_class']}
        Confidence: {record_data['confidence']:.1%}
        
        Complete Blood Count (CBC) Values:
        - White Blood Cell Count (WBC): {record_data['wbc']:.2f} 10³/µL
        - Red Blood Cell Count (RBC): {record_data['rbc']:.2f} million/µL
        - Hemoglobin (HGB): {record_data['hgb']:.2f} g/dL
        - Hematocrit (HCT): {record_data['hct']:.2f} %
        - Mean Corpuscular Volume (MCV): {record_data['mcv']:.2f} fL
        - Mean Corpuscular Hemoglobin (MCH): {record_data['mch']:.2f} pg
        - Mean Corpuscular Hemoglobin Concentration (MCHC): {record_data['mchc']:.2f} g/dL
        - Platelet Count (PLT): {record_data['plt']:.2f} 10³/µL
        
        {f'Notes: {record_data.get("notes", "")}' if record_data.get("notes") else ''}
        
        Important Information:
        Please note: This is an AI-powered screening tool and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
        
        Test Date: {record_data['created_at']}
        
        This email was sent from AnemoCheck - Anemia Detection System
        For support, please contact your healthcare provider
        """
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = smtp_username
        msg['To'] = user_email
        
        # Add text and HTML parts
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            # Try different authentication methods for Outlook
            try:
                server.login(smtp_username, smtp_password)
            except smtplib.SMTPAuthenticationError as e:
                if "basic authentication is disabled" in str(e).lower():
                    # For Outlook, try with different approach
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(smtp_username, smtp_password)
                else:
                    raise e
            server.send_message(msg)
        
        return True, "Email sent successfully"
        
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False, f"Error sending email: {str(e)}"


# Email routes
@app.route('/send-result-email/<int:record_id>', methods=['POST'])
@login_required
def send_result_email(record_id):
    """Send anemia test result email to current user."""
    try:
        logger.info(f"Attempting to send email for record {record_id} to user {current_user.id}")
        
        # Get record data
        record_data = db.get_classification_record(record_id)
        if not record_data:
            logger.error(f"Record {record_id} not found")
            return jsonify({'success': False, 'error': 'Record not found'})
        
        logger.info(f"Found record: {record_data}")
        
        # Check if user owns this record
        if record_data['user_id'] != current_user.id:
            logger.error(f"User {current_user.id} does not own record {record_id}")
            return jsonify({'success': False, 'error': 'Access denied'})
        
        # Get user data
        user_data = db.get_user_by_id(current_user.id)
        if not user_data:
            logger.error(f"User {current_user.id} not found")
            return jsonify({'success': False, 'error': 'User not found'})
        
        logger.info(f"Found user: {user_data['email']}")
        
        # Send email
        success, message = send_result_email_helper(record_id, user_data['email'], 
                                           f"{user_data['first_name']} {user_data['last_name']}".strip() or user_data['username'],
                                           record_data)
        
        if success:
            logger.info(f"Email sent successfully to {user_data['email']}")
            return jsonify({'success': True, 'message': message, 'email': user_data['email']})
        else:
            logger.error(f"Email sending failed: {message}")
            return jsonify({'success': False, 'error': message})
            
    except Exception as e:
        logger.error(f"Error in send_result_email route: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'})


@app.route('/admin/send-result-email/<int:record_id>', methods=['POST'])
@login_required
def admin_send_result_email(record_id):
    """Send anemia test result email (admin function)."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Admin access required'})
    
    try:
        logger.info(f"Admin {current_user.id} attempting to send email for record {record_id}")
        
        # Get record data
        record_data = db.get_classification_record(record_id)
        if not record_data:
            logger.error(f"Record {record_id} not found")
            return jsonify({'success': False, 'error': 'Record not found'})
        
        # Get user data
        user_data = db.get_user_by_id(record_data['user_id'])
        if not user_data:
            logger.error(f"User {record_data['user_id']} not found")
            return jsonify({'success': False, 'error': 'User not found'})
        
        logger.info(f"Found user: {user_data['email']}")
        
        # Send email
        success, message = send_result_email_helper(record_id, user_data['email'], 
                                           f"{user_data['first_name']} {user_data['last_name']}".strip() or user_data['username'],
                                           record_data)
        
        if success:
            logger.info(f"Email sent successfully to {user_data['email']}")
            return jsonify({'success': True, 'message': message, 'email': user_data['email']})
        else:
            logger.error(f"Email sending failed: {message}")
            return jsonify({'success': False, 'error': message})
            
    except Exception as e:
        logger.error(f"Error in admin_send_result_email route: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'})


# Export routes
@app.route('/admin/export/dashboard.csv')
@login_required
def export_dashboard():
    """Export dashboard statistics as CSV."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get system statistics
    stats = db.get_statistics()
    
    # Create CSV content with proper formatting
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Metric', 'Value'])
    
    # Write statistics
    writer.writerow(['Total Users', stats['total_users']])
    writer.writerow(['Total Classifications', stats['total_classifications']])
    writer.writerow(['Anemic Cases', stats['anemic_cases']])
    writer.writerow(['Normal Cases', stats['normal_cases']])
    writer.writerow(['New Users (Last 7 Days)', stats['new_user_count']])
    writer.writerow(['Active Users (Last 7 Days)', stats['active_user_count']])
    
    # Add empty row
    writer.writerow([])
    writer.writerow(['Classification Distribution'])
    
    # Add class distribution
    for class_name, count in stats['class_distribution'].items():
        writer.writerow([class_name, count])
    
    csv_content = output.getvalue()
    output.close()
    
    # Create response with BOM for proper Excel compatibility
    from flask import Response
    # Add BOM for proper UTF-8 handling in Excel
    csv_content_with_bom = '\ufeff' + csv_content
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=dashboard_statistics.csv'}
    )
    return response


@app.route('/admin/export/classification_stats.csv')
@login_required
def export_classification_stats():
    """Export classification statistics as CSV."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get system statistics and chart data
    stats = db.get_statistics()
    charts_data = db.get_admin_dashboard_charts()
    
    # Create CSV content with proper formatting
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Classification Statistics Export'])
    writer.writerow(['Generated on', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])
    
    # Write overall statistics
    writer.writerow(['Overall Statistics'])
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Total Users', stats['total_users']])
    writer.writerow(['Total Classifications', stats['total_classifications']])
    writer.writerow(['Anemic Cases', stats['anemic_cases']])
    writer.writerow(['Normal Cases', stats['normal_cases']])
    writer.writerow([])
    
    # Write age group distribution
    writer.writerow(['Age Group Distribution'])
    writer.writerow(['Age Group', 'Count'])
    for age_group, count in charts_data['age_groups'].items():
        writer.writerow([age_group, count])
    writer.writerow([])
    
    # Write gender distribution
    writer.writerow(['Gender Distribution'])
    writer.writerow(['Gender', 'Count'])
    for gender, count in charts_data['gender_stats'].items():
        writer.writerow([gender, count])
    writer.writerow([])
    
    # Write severity classification distribution
    writer.writerow(['Severity Classification Distribution'])
    writer.writerow(['Severity', 'Count'])
    for severity, count in charts_data['severity_stats'].items():
        writer.writerow([severity, count])
    
    csv_content = output.getvalue()
    output.close()
    
    # Create response with BOM for proper Excel compatibility
    from flask import Response
    # Add BOM for proper UTF-8 handling in Excel
    csv_content_with_bom = '\ufeff' + csv_content
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=classification_statistics.csv'}
    )
    return response


@app.route('/admin/import/classification_data', methods=['POST'])
@csrf.exempt
@login_required
def import_classification_data():
    """Import classification data from CSV file."""
    try:
        if not current_user.is_admin:
            return jsonify({'success': False, 'error': 'Access denied. Administrator privileges required.'})
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'File must be a CSV file'})
        
        import csv
        import io
        
        # Read CSV content
        csv_content = file.read().decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        # Expected columns: age, gender, category
        required_columns = ['age', 'gender', 'category']
        if not all(col in csv_reader.fieldnames for col in required_columns):
            return jsonify({
                'success': False, 
                'error': f'CSV must contain columns: {", ".join(required_columns)}'
            })
        
        imported_count = 0
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        # Ensure the table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_import_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                category TEXT NOT NULL,
                imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        for row in csv_reader:
            try:
                age = int(row['age'])
                gender = row['gender'].strip()
                category = row['category'].strip()
                
                # Insert into the table for statistics
                cursor.execute('''
                    INSERT INTO classification_import_data (age, gender, category, imported_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (age, gender, category))
                
                imported_count += 1
            except (ValueError, KeyError) as e:
                continue  # Skip invalid rows
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'imported_count': imported_count,
            'message': f'Successfully imported {imported_count} records'
        })
        
    except Exception as e:
        logger.error(f"Import error: {str(e)}")
        return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'})


@app.route('/admin/api/charts_data')
@login_required
def get_charts_data_api():
    """API endpoint to get updated chart data."""
    try:
        if not current_user.is_admin:
            return jsonify({'success': False, 'error': 'Access denied'})
        
        # Get base chart data
        charts_data = db.get_admin_dashboard_charts()
        
        # Get imported data and merge with existing data
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        # Ensure the table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_import_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                category TEXT NOT NULL,
                imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Age groups from imported data
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN age < 18 THEN 'Under 18'
                    WHEN age BETWEEN 18 AND 30 THEN '18-30'
                    WHEN age BETWEEN 31 AND 45 THEN '31-45'
                    WHEN age BETWEEN 46 AND 60 THEN '46-60'
                    ELSE 'Over 60'
                END as age_group,
                COUNT(*) as count
            FROM classification_import_data
            GROUP BY age_group
        ''')
        imported_age_groups = {row['age_group']: row['count'] for row in cursor.fetchall()}
        
        # Gender stats from imported data
        cursor.execute('''
            SELECT gender, COUNT(*) as count
            FROM classification_import_data
            GROUP BY gender
        ''')
        imported_gender_stats = {row['gender']: row['count'] for row in cursor.fetchall()}
        
        # Severity stats from imported data
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM classification_import_data
            GROUP BY category
        ''')
        imported_severity_stats = {row['category']: row['count'] for row in cursor.fetchall()}
        
        conn.close()
        
        # Merge imported data with existing data
        for age_group, count in imported_age_groups.items():
            charts_data['age_groups'][age_group] = charts_data['age_groups'].get(age_group, 0) + count
        
        # Normalize and merge gender data
        for gender, count in imported_gender_stats.items():
            # Normalize gender to match existing format
            normalized_gender = gender.title() if gender else 'Other'
            charts_data['gender_stats'][normalized_gender] = charts_data['gender_stats'].get(normalized_gender, 0) + count
        
        # Normalize and merge severity data
        for category, count in imported_severity_stats.items():
            # Normalize category to match existing format
            normalized_category = normalize_severity_category(category)
            charts_data['severity_stats'][normalized_category] = charts_data['severity_stats'].get(normalized_category, 0) + count
        
        return jsonify({'success': True, 'data': charts_data})
        
    except Exception as e:
        logger.error(f"Charts data API error: {str(e)}")
        return jsonify({'success': False, 'error': f'Error fetching chart data: {str(e)}'})


def normalize_severity_category(category):
    """Normalize severity category names to standard format."""
    if not category:
        return 'Other'
    
    category_lower = category.lower().strip()
    
    if 'normal' in category_lower:
        return 'Normal'
    elif 'mild' in category_lower and 'anemia' in category_lower:
        return 'Mild Anemia'
    elif 'mild' in category_lower:
        return 'Mild Anemia'
    elif 'moderate' in category_lower and 'anemia' in category_lower:
        return 'Moderate Anemia'
    elif 'moderate' in category_lower:
        return 'Moderate Anemia'
    elif 'severe' in category_lower and 'anemia' in category_lower:
        return 'Severe Anemia'
    elif 'severe' in category_lower:
        return 'Severe Anemia'
    else:
        return category.title()  # Return as-is if no match


@app.route('/admin/export/users.csv')
@login_required
def export_users():
    """Export users data as CSV."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get all users
    users = db.get_all_users()
    
    # Create CSV content with proper formatting
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Username', 'Email', 'First Name', 'Last Name', 'Gender', 'Date of Birth', 'Medical ID', 'Is Admin', 'Created At', 'Last Login'])
    
    # Write user data
    for user in users:
        writer.writerow([
            user['id'],
            user['username'],
            user['email'],
            user['first_name'] or '',
            user['last_name'] or '',
            user['gender'] or '',
            user['date_of_birth'] or '',
            user['medical_id'] or '',
            'Yes' if user['is_admin'] else 'No',
            user['created_at'],
            user['last_login'] or ''
        ])
    
    csv_content = output.getvalue()
    output.close()
    
    # Create response with BOM for proper Excel compatibility
    from flask import Response
    # Add BOM for proper UTF-8 handling in Excel
    csv_content_with_bom = '\ufeff' + csv_content
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=users_export.csv'}
    )
    return response


@app.route('/admin/export/classification_history.csv')
@login_required
def export_classification_history():
    """Export classification history as CSV."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get all classification history
    records = db.get_all_classification_history()
    
    # Create CSV content with proper formatting
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'User ID', 'Username', 'Date', 'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 
                    'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophil', 'Immature Granulocytes', 
                    'Predicted Class', 'Confidence', 'Recommendation', 'Notes'])
    
    # Write record data
    for record in records:
        writer.writerow([
            record['id'],
            record['user_id'],
            record['username'],
            record['created_at'],
            record['wbc'],
            record['rbc'],
            record['hgb'],
            record['hct'],
            record['mcv'],
            record['mch'],
            record['mchc'],
            record['plt'],
            record['neutrophils'] or '',
            record['lymphocytes'] or '',
            record['monocytes'] or '',
            record['eosinophils'] or '',
            record['basophil'] or '',
            record['immature_granulocytes'] or '',
            record['predicted_class'],
            record['confidence'],
            record['recommendation'] or '',
            record['notes'] or ''
        ])
    
    csv_content = output.getvalue()
    output.close()
    
    # Create response with BOM for proper Excel compatibility
    from flask import Response
    # Add BOM for proper UTF-8 handling in Excel
    csv_content_with_bom = '\ufeff' + csv_content
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=classification_history.csv'}
    )
    return response


# Classification History Actions
@app.route('/admin/classification/<int:record_id>/details')
@login_required
def admin_classification_details(record_id):
    """Get detailed classification information for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get classification record
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ch.*, u.username, u.first_name, u.last_name, u.email
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE ch.id = ?
    """, (record_id,))
    
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    # Convert to dict
    record = dict(record)
    
    # Prepare response data
    classification_details = {
        'record': record,
        'user': {
            'username': record['username'],
            'first_name': record['first_name'],
            'last_name': record['last_name'],
            'email': record['email']
        }
    }
    
    return jsonify({'success': True, 'data': classification_details})


@app.route('/admin/classification/<int:record_id>/edit', methods=['GET', 'POST'])
@login_required
def admin_edit_classification(record_id):
    """Edit classification record."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('admin_history'))
    
    # Get classification record
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ch.*, u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE ch.id = ?
    """, (record_id,))
    
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        flash('Record not found.')
        return redirect(url_for('admin_history'))
    
    record = dict(record)
    
    if request.method == 'POST':
        # Update the record
        wbc = request.form.get('wbc', type=float)
        rbc = request.form.get('rbc', type=float)
        hgb = request.form.get('hgb', type=float)
        hct = request.form.get('hct', type=float)
        mcv = request.form.get('mcv', type=float)
        mch = request.form.get('mch', type=float)
        mchc = request.form.get('mchc', type=float)
        plt = request.form.get('plt', type=float)
        neutrophils = request.form.get('neutrophils', type=float)
        lymphocytes = request.form.get('lymphocytes', type=float)
        monocytes = request.form.get('monocytes', type=float)
        eosinophils = request.form.get('eosinophils', type=float)
        basophil = request.form.get('basophil', type=float)
        immature_granulocytes = request.form.get('immature_granulocytes', type=float)
        predicted_class = request.form.get('predicted_class')
        confidence_percentage = request.form.get('confidence', type=float)
        
        # Validate confidence percentage
        if confidence_percentage is None or confidence_percentage < 0 or confidence_percentage > 100:
            flash('Confidence must be between 0 and 100%.')
            return render_template('admin/edit_classification.html', record=record)
        
        # Convert percentage (0-100) to decimal (0-1) for storage
        confidence = confidence_percentage / 100.0
        recommendation = request.form.get('recommendation')
        notes = request.form.get('notes')
        
        # Update the record in database
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE classification_history 
            SET wbc = ?, rbc = ?, hgb = ?, hct = ?, mcv = ?, mch = ?, mchc = ?, plt = ?,
                neutrophils = ?, lymphocytes = ?, monocytes = ?, eosinophils = ?, basophil = ?, immature_granulocytes = ?,
                predicted_class = ?, confidence = ?, recommendation = ?, notes = ?
            WHERE id = ?
        """, (wbc, rbc, hgb, hct, mcv, mch, mchc, plt, neutrophils, lymphocytes, monocytes, 
              eosinophils, basophil, immature_granulocytes, predicted_class, confidence, 
              recommendation, notes, record_id))
        
        conn.commit()
        conn.close()
        
        flash('Classification record updated successfully!')
        return redirect(url_for('admin_history'))
    
    return render_template('admin/edit_classification.html', record=record)


@app.route('/admin/classification/<int:record_id>/delete', methods=['POST'])
@login_required
def admin_delete_classification(record_id):
    """Delete classification record."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get record details for confirmation
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ch.*, u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE ch.id = ?
    """, (record_id,))
    
    record = cursor.fetchone()
    
    if not record:
        conn.close()
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    # Delete the record
    cursor.execute("DELETE FROM classification_history WHERE id = ?", (record_id,))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Admin {current_user.username} deleted classification record ID {record_id}")
    return jsonify({'success': True, 'message': f'Classification record for {record["username"]} deleted successfully'})


@app.route('/admin/classification/filtered-data')
@login_required
def admin_classification_filtered_data():
    """Get filtered classification history data for AJAX requests."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get filter parameters
    user_filter = request.args.get('user', '')
    result_filter = request.args.get('result', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    # Build query
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    # Base query
    query = """
        SELECT ch.*, u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE 1=1
    """
    params = []
    
    # Apply user filter
    if user_filter:
        query += " AND u.username = ?"
        params.append(user_filter)
    
    # Apply result filter
    if result_filter:
        if result_filter == 'Anemic':
            # Filter for all anemic types (Mild, Moderate, Severe)
            query += " AND ch.predicted_class IN ('Mild', 'Moderate', 'Severe')"
        else:
            query += " AND ch.predicted_class = ?"
            params.append(result_filter)
    
    # Apply date filters
    if date_from:
        query += " AND DATE(ch.created_at) >= ?"
        params.append(date_from)
    
    if date_to:
        query += " AND DATE(ch.created_at) <= ?"
        params.append(date_to)
    
    # Order by date descending
    query += " ORDER BY ch.created_at DESC"
    
    cursor.execute(query, params)
    records = cursor.fetchall()
    conn.close()
    
    # Convert to list of dicts
    filtered_records = []
    for record in records:
        record_dict = dict(record)
        # Format confidence as percentage
        record_dict['confidence_percentage'] = round(record_dict['confidence'] * 100, 2)
        filtered_records.append(record_dict)
    
    return jsonify({
        'success': True,
        'data': filtered_records,
        'total_count': len(filtered_records)
    })


@app.route('/admin/classification/available-users')
@login_required
def admin_classification_available_users():
    """Get list of users who have classification records."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    # Get unique usernames from classification history
    cursor.execute("""
        SELECT DISTINCT u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        ORDER BY u.username
    """)
    
    users = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({
        'success': True,
        'users': users
    })


# Simple Chat routes
simple_chat.init_chat_tables()

@app.route('/chat')
@login_required
def chat():
    """User messenger interface with admin chat."""
    # Get user conversations
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=False)
    
    return render_template('user_messenger.html', 
                         conversations=conversations,
                         current_user=current_user)

@app.route('/admin/messenger')
@login_required
def admin_messenger():
    """Admin messenger interface with individual chat windows."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get all users
    all_users = simple_chat.get_all_users()
    logger.info(f"Admin messenger: Found {len(all_users)} users")
    
    # Get admin conversations for history
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=True)
    logger.info(f"Admin messenger: Found {len(conversations)} conversations")
    
    return render_template('admin/messenger.html', 
                         all_users=all_users,
                         conversations=conversations,
                         current_user=current_user)

@app.route('/admin/chat/start', methods=['POST'])
@login_required
def admin_start_chat():
    """Admin starts chat with user."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    data = request.get_json()
    user_id = data.get('user_id')
    logger.info(f"Admin start chat: user_id={user_id}")
    
    if not user_id:
        return jsonify({'success': False, 'error': 'User ID required'}), 400
    
    # Create conversation
    success, conversation_id = simple_chat.create_conversation(user_id, admin_id=current_user.id)
    logger.info(f"Admin start chat: success={success}, conversation_id={conversation_id}")
    
    if success:
        return jsonify({
            'success': True,
            'conversation_id': conversation_id
        })
    else:
        return jsonify({'success': False, 'error': conversation_id}), 500

@app.route('/admin/chat/conversation/<int:user_id>')
@login_required
def admin_get_conversation(user_id):
    """Get conversation between admin and user."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Find existing conversation
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=True)
    conversation = next((c for c in conversations if c['user_id'] == user_id), None)
    
    if conversation:
        return jsonify({
            'success': True,
            'conversation_id': conversation['id']
        })
    else:
        return jsonify({'success': False, 'error': 'No conversation found'})

@app.route('/admin/chat/messages/<int:conversation_id>')
@login_required
def admin_get_messages(conversation_id):
    """Get messages for conversation."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    logger.info(f"Admin get messages: conversation_id={conversation_id}, admin_id={current_user.id}")
    
    messages = simple_chat.get_conversation_messages(conversation_id)
    logger.info(f"Admin get messages result: {len(messages)} messages found")
    
    return jsonify({
        'success': True,
        'messages': messages
    })

@app.route('/admin/chat/delete-message', methods=['POST'])
@login_required
def admin_delete_message():
    """Delete a specific message for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    data = request.get_json()
    message_id = data.get('message_id')
    
    logger.info(f"Admin delete message: message_id={message_id}, admin_id={current_user.id}")
    
    if not message_id:
        return jsonify({'success': False, 'error': 'Message ID required'}), 400
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Check if message belongs to current admin
        cursor.execute('SELECT sender_id FROM chat_messages WHERE id = ?', (message_id,))
        message = cursor.fetchone()
        
        if not message:
            return jsonify({'success': False, 'error': 'Message not found'}), 404
        
        if message['sender_id'] != current_user.id:
            return jsonify({'success': False, 'error': 'You can only delete your own messages'}), 403
        
        # Delete the message
        cursor.execute('DELETE FROM chat_messages WHERE id = ?', (message_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Admin message deleted: message_id={message_id}")
        return jsonify({'success': True, 'message': 'Message deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting admin message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/chat/delete-conversation', methods=['POST'])
@login_required
def admin_delete_conversation():
    """Delete entire conversation for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    logger.info(f"Admin delete conversation: conversation_id={conversation_id}, admin_id={current_user.id}")
    
    if not conversation_id:
        return jsonify({'success': False, 'error': 'Conversation ID required'}), 400
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Delete all messages from this conversation
        cursor.execute('DELETE FROM chat_messages WHERE conversation_id = ?', (conversation_id,))
        
        # Delete the conversation itself
        cursor.execute('DELETE FROM chat_conversations WHERE id = ?', (conversation_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Admin conversation deleted: conversation_id={conversation_id}")
        return jsonify({'success': True, 'message': 'Conversation deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting admin conversation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/chat/send', methods=['POST'])
@login_required
def admin_send_message():
    """Admin sends message."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    message_text = data.get('message', '').strip()
    
    logger.info(f"Admin send message: conversation_id={conversation_id}, message_text='{message_text}', admin_id={current_user.id}")
    
    if not message_text:
        return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
    
    if not conversation_id:
        return jsonify({'success': False, 'error': 'Conversation ID required'}), 400
    
    # Send message
    success, message_id = simple_chat.send_message(conversation_id, current_user.id, message_text)
    logger.info(f"Admin send message result: success={success}, message_id={message_id}")
    
    if success:
        return jsonify({'success': True, 'message_id': message_id})
    else:
        return jsonify({'success': False, 'error': message_id}), 500

@app.route('/user/chat/start', methods=['POST'])
@login_required
def user_start_chat():
    """User starts chat with admin."""
    data = request.get_json()
    admin_id = data.get('admin_id', 1)  # Default to admin ID 1 if not specified
    
    # Check if conversation already exists with this admin
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=False)
    existing_conversation = None
    
    for conv in conversations:
        if conv.get('admin_id') == admin_id:
            existing_conversation = conv
            break
    
    if existing_conversation:
        # Return existing conversation
        return jsonify({
            'success': True,
            'conversation_id': existing_conversation['id']
        })
    else:
        # Create new conversation with specific admin
        success, conversation_id = simple_chat.create_conversation(current_user.id, admin_id)
        
        if success:
            return jsonify({
                'success': True,
                'conversation_id': conversation_id
            })
        else:
            return jsonify({'success': False, 'error': conversation_id}), 500

@app.route('/user/chat/conversation')
@login_required
def user_get_conversation():
    """Get user's conversation with admin."""
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=False)
    
    if conversations:
        return jsonify({
            'success': True,
            'conversation_id': conversations[0]['id']
        })
    else:
        return jsonify({'success': False, 'error': 'No conversation found'})

@app.route('/user/chat/messages/<int:conversation_id>')
@login_required
def user_get_messages(conversation_id):
    """Get messages for user conversation."""
    logger.info(f"User get messages: conversation_id={conversation_id}, user_id={current_user.id}")
    
    messages = simple_chat.get_conversation_messages(conversation_id)
    logger.info(f"User get messages result: {len(messages)} messages found")
    
    return jsonify({
        'success': True,
        'messages': messages
    })

@app.route('/user/chat/clear-history', methods=['POST'])
@login_required
def user_clear_chat_history():
    """Clear chat history for user."""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    logger.info(f"User clear chat history: conversation_id={conversation_id}, user_id={current_user.id}")
    
    if not conversation_id:
        return jsonify({'success': False, 'error': 'Conversation ID required'}), 400
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Delete all messages from this conversation
        cursor.execute('DELETE FROM chat_messages WHERE conversation_id = ?', (conversation_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User chat history cleared: conversation_id={conversation_id}")
        return jsonify({'success': True, 'message': 'Chat history cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/user/chat/delete-message', methods=['POST'])
@login_required
def user_delete_message():
    """Delete a specific message for user."""
    data = request.get_json()
    message_id = data.get('message_id')
    
    logger.info(f"User delete message: message_id={message_id}, user_id={current_user.id}")
    
    if not message_id:
        return jsonify({'success': False, 'error': 'Message ID required'}), 400
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Check if message belongs to current user
        cursor.execute('SELECT sender_id FROM chat_messages WHERE id = ?', (message_id,))
        message = cursor.fetchone()
        
        if not message:
            return jsonify({'success': False, 'error': 'Message not found'}), 404
        
        if message['sender_id'] != current_user.id:
            return jsonify({'success': False, 'error': 'You can only delete your own messages'}), 403
        
        # Delete the message
        cursor.execute('DELETE FROM chat_messages WHERE id = ?', (message_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User message deleted: message_id={message_id}")
        return jsonify({'success': True, 'message': 'Message deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/user/chat/send', methods=['POST'])
@login_required
def user_send_message():
    """User sends message."""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    message_text = data.get('message', '').strip()
    
    logger.info(f"User send message: conversation_id={conversation_id}, message_text='{message_text}', user_id={current_user.id}")
    
    if not message_text:
        return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
    
    if not conversation_id:
        return jsonify({'success': False, 'error': 'Conversation ID required'}), 400
    
    # Send message
    success, message_id = simple_chat.send_message(conversation_id, current_user.id, message_text)
    logger.info(f"User send message result: success={success}, message_id={message_id}")
    
    if success:
        return jsonify({'success': True, 'message_id': message_id})
    else:
        return jsonify({'success': False, 'error': message_id}), 500

@app.route('/admin/chat/clear-data', methods=['POST'])
@login_required
def admin_clear_chat_data():
    """Clear all chat data (for testing purposes)."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Clear all messages
        cursor.execute('DELETE FROM chat_messages')
        
        # Clear all conversations
        cursor.execute('DELETE FROM chat_conversations')
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Chat data cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/chat/check-new-messages')
@login_required
def admin_check_new_messages():
    """Check for new messages for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get conversations where admin is involved
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=True)
    
    new_messages = []
    for conv in conversations:
        # Get latest message
        messages = simple_chat.get_conversation_messages(conv['id'])
        if messages:
            latest_message = messages[-1]
            # Check if message is from user (not admin) and recent
            if latest_message['sender_id'] != current_user.id:
                # Check if message is within last 5 minutes
                message_time = datetime.fromisoformat(latest_message['created_at'].replace('Z', '+00:00'))
                if (datetime.now() - message_time).seconds < 300:  # 5 minutes
                    new_messages.append({
                        'user_id': conv['user_id'],
                        'username': conv['username'],
                        'message': latest_message['message_text'][:50] + '...' if len(latest_message['message_text']) > 50 else latest_message['message_text']
                    })
    
    return jsonify({
        'success': True,
        'new_messages': new_messages
    })

@app.route('/chat/unread-count')
@login_required
def get_unread_count():
    """Get unread message count."""
    # For simple chat, we'll return 0 as we don't track unread status
    return jsonify({
        'success': True,
        'unread_count': 0
    })


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection to WebSocket."""
    if current_user.is_authenticated:
        # Join a room specific to this user
        join_room(str(current_user.id))
        logger.info(f"User {current_user.username} connected to WebSocket")
        
        # If admin, also join admin room
        if current_user.is_admin:
            join_room('admin_room')
            logger.info(f"Admin {current_user.username} joined admin room")
    else:
        logger.info("Anonymous user connected to WebSocket")


@socketio.on('request_update')
def handle_update_request():
    """Handle client request for updates."""
    if current_user.is_authenticated:
        # Get user's recent classification history
        history = db.get_user_classification_history(current_user.id, limit=5)
        emit('history_update', history)


@socketio.on('join_conversation')
def handle_join_conversation(data):
    """Handle user joining a conversation room."""
    if current_user.is_authenticated:
        conversation_id = data.get('conversation_id')
        if conversation_id:
            # Verify user has access to this conversation
            conversations = simple_chat.get_user_conversations(current_user.id, current_user.is_admin)
            if any(conv['id'] == conversation_id for conv in conversations):
                join_room(f'conversation_{conversation_id}')
                logger.info(f"User {current_user.username} joined conversation {conversation_id}")


@socketio.on('leave_conversation')
def handle_leave_conversation(data):
    """Handle user leaving a conversation room."""
    if current_user.is_authenticated:
        conversation_id = data.get('conversation_id')
        if conversation_id:
            leave_room(f'conversation_{conversation_id}')
            logger.info(f"User {current_user.username} left conversation {conversation_id}")


@socketio.on('typing')
def handle_typing(data):
    """Handle typing indicator."""
    if current_user.is_authenticated:
        conversation_id = data.get('conversation_id')
        is_typing = data.get('is_typing', False)
        
        if conversation_id:
            typing_data = {
                'user_id': current_user.id,
                'username': current_user.username,
                'is_typing': is_typing
            }
            emit('user_typing', typing_data, room=f'conversation_{conversation_id}', include_self=False)


# Initialize the database and model when the app starts
with app.app_context():
    # Initialize database if it doesn't exist
    if not os.path.exists(db.DB_PATH):
        db.init_db()
    
    # Initialize anemia model with system settings
    threshold_normal = float(db.get_system_setting('threshold_normal') or 12.0)
    threshold_mild = float(db.get_system_setting('threshold_mild') or 10.0)
    threshold_moderate = float(db.get_system_setting('threshold_moderate') or 8.0)
    model_type = db.get_system_setting('model_type') or 'decision_tree'
    
    anemia_model.update_thresholds(
        threshold_normal=threshold_normal,
        threshold_mild=threshold_mild,
        threshold_moderate=threshold_moderate
    )
    anemia_model.set_model_type(model_type)
    #anemia_model.initialize()


@app.route('/export/history.csv')
@login_required
def export_my_classification_history():
    """Export current user's classification history as CSV."""
    # Fetch current user's records
    records = db.get_user_classification_history(current_user.id, limit=100000)

    # Build CSV
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header similar to user table view
    writer.writerow(['Date', 'WBC', 'RBC', 'HGB (g/dL)', 'HCT (%)', 'MCV (fL)', 'MCH (pg)', 'MCHC (g/dL)', 'PLT',
                     'NEU (%)', 'LYM (%)', 'MON (%)', 'EOS (%)', 'BAS (%)', 'IGR (%)', 'Classification', 'Confidence', 'Notes'])

    for record in records:
        writer.writerow([
            record.get('created_at', ''),
            record.get('wbc', ''),
            record.get('rbc', ''),
            record.get('hgb', ''),
            record.get('hct', ''),
            record.get('mcv', ''),
            record.get('mch', ''),
            record.get('mchc', ''),
            record.get('plt', ''),
            record.get('neutrophils') if record.get('neutrophils') is not None else '',
            record.get('lymphocytes') if record.get('lymphocytes') is not None else '',
            record.get('monocytes') if record.get('monocytes') is not None else '',
            record.get('eosinophils') if record.get('eosinophils') is not None else '',
            record.get('basophil') if record.get('basophil') is not None else '',
            record.get('immature_granulocytes') if record.get('immature_granulocytes') is not None else '',
            record.get('predicted_class', ''),
            f"{round(float(record.get('confidence', 0))*100)}%" if record.get('confidence') is not None else '',
            record.get('notes', '') or ''
        ])

    csv_content = output.getvalue()
    output.close()

    # Response with BOM for Excel compatibility
    from flask import Response
    csv_content_with_bom = '\ufeff' + csv_content
    filename = f"classification_history_{current_user.username}.csv"
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )
    return response


@app.route('/api/profile/email-exists')
@login_required
def api_profile_email_exists():
    """Check if an email already exists (excluding current user's email)."""
    email = (request.args.get('email') or '').strip()
    if not email:
        return jsonify({ 'success': True, 'exists': False, 'isCurrent': False })
    user = db.get_user_by_email(email)
    if not user:
        return jsonify({ 'success': True, 'exists': False, 'isCurrent': False })
    # If found but it's the same as current user, allow it
    is_current = (str(user.get('id')) == str(current_user.id))
    return jsonify({ 'success': True, 'exists': not is_current, 'isCurrent': is_current })

@app.route('/api/profile/medical-id-exists')
@login_required
def api_profile_medical_id_exists():
    """Check if a medical ID already exists (excluding current user's own)."""
    mid = (request.args.get('medical_id') or '').strip()
    # Empty medical_id is allowed (treated as NULL) — never considered a duplicate
    if not mid:
        return jsonify({ 'success': True, 'exists': False, 'isCurrent': False })
    user = db.get_user_by_medical_id(mid)
    if not user:
        return jsonify({ 'success': True, 'exists': False, 'isCurrent': False })
    is_current = (str(user.get('id')) == str(current_user.id))
    return jsonify({ 'success': True, 'exists': not is_current, 'isCurrent': is_current })


@app.route('/api/register/username-exists')
def api_register_username_exists():
    """Public endpoint: check if a username already exists for registration page."""
    username = (request.args.get('username') or '').strip()
    if not username:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_username(username)
    return jsonify({ 'success': True, 'exists': bool(user) })

@app.route('/api/register/email-exists')
def api_register_email_exists():
    """Public endpoint: check if an email already exists for registration page."""
    email = (request.args.get('email') or '').strip()
    if not email:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_email(email)
    return jsonify({ 'success': True, 'exists': bool(user) })

@app.route('/api/register/medical-id-exists')
def api_register_medical_id_exists():
    """Public endpoint: check if a medical ID already exists for registration page."""
    medical_id = (request.args.get('medical_id') or '').strip()
    if not medical_id:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_medical_id(medical_id)
    return jsonify({ 'success': True, 'exists': bool(user) })


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)