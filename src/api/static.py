from flask import send_from_directory
import os

def init_static_routes(app):
    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename) 