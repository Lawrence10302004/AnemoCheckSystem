#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forms for Anemia Detection System
--------------------------------
This module defines the forms used in the application.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, FloatField, SelectField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, NumberRange
from models import User

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
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        """Validate that username is unique."""
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')
    
    def validate_email(self, email):
        """Validate that email is unique."""
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

class CBCForm(FlaskForm):
    """Form for CBC data input."""
    gender = SelectField('Gender', choices=[('female', 'Female'), ('male', 'Male')], validators=[DataRequired()])
    hgb = FloatField('Hemoglobin (HGB) in g/dL', validators=[
        DataRequired(), 
        NumberRange(min=1, max=25, message='Please enter a valid value between 1 and 25')
    ])
    rbc = FloatField('Red Blood Cell Count (RBC) in million cells/mcL', validators=[
        DataRequired(), 
        NumberRange(min=0.5, max=10, message='Please enter a valid value between 0.5 and 10')
    ])
    hct = FloatField('Hematocrit (HCT) in %', validators=[
        DataRequired(), 
        NumberRange(min=5, max=70, message='Please enter a valid value between 5 and 70')
    ])
    mcv = FloatField('Mean Corpuscular Volume (MCV) in fL', validators=[
        DataRequired(), 
        NumberRange(min=50, max=150, message='Please enter a valid value between 50 and 150')
    ])
    source = SelectField('Source', choices=[
        ('lab_test', 'Laboratory Test'),
        ('self_reported', 'Self-Reported'),
        ('other', 'Other')
    ])
    notes = TextAreaField('Notes', validators=[Length(max=500)])
    submit = SubmitField('Detect Anemia')

class ProfileForm(FlaskForm):
    """Form for updating user profile."""
    first_name = StringField('First Name', validators=[Length(max=64)])
    last_name = StringField('Last Name', validators=[Length(max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')])
    current_password = PasswordField('Current Password')
    new_password = PasswordField('New Password', validators=[Length(min=8)])
    confirm_password = PasswordField('Confirm New Password', validators=[EqualTo('new_password')])
    submit = SubmitField('Update Profile')