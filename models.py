#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database Models for Anemia Detection System
------------------------------------------
This module defines the database models for the application.
"""

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db

class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    gender = db.Column(db.String(10))
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship with CBCRecord
    cbc_records = db.relationship('CBCRecord', backref='user', lazy='dynamic', 
                                cascade="all, delete-orphan")
    
    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class CBCRecord(db.Model):
    """Model for storing CBC test records."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # CBC Values
    hgb = db.Column(db.Float, nullable=False)  # Hemoglobin
    rbc = db.Column(db.Float, nullable=False)  # Red Blood Cell count
    hct = db.Column(db.Float, nullable=False)  # Hematocrit
    mcv = db.Column(db.Float, nullable=False)  # Mean Corpuscular Volume
    
    # Prediction Results
    result = db.Column(db.String(20))  # 'Anemic' or 'Normal'
    probability = db.Column(db.Float)  # Probability of anemia
    
    # Additional fields
    notes = db.Column(db.Text)
    source = db.Column(db.String(50))  # e.g., 'Lab Test', 'Self-reported'
    
    def __repr__(self):
        return f'<CBCRecord {self.id} - User {self.user_id}>'