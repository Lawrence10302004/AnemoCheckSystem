"""
Main entry point for the AnemoCheck Flask application.
This file imports the app and socketio from app.py to be used by Gunicorn.
"""

from app import app, socketio
#import rfc_anemia_classifier_training

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")  # auto-open browser
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
