import os
from flask import Flask, request, make_response, Response, Blueprint
import json
import time
from dotenv import load_dotenv
from flask_cors import CORS

class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.app_bp = Blueprint('app', __name__)
        self._setup_routes()
        self._initialize_app()
        CORS(self.app)
    
    def _initialize_app(self):
        with self.app.app_context():
            load_dotenv()
            if self.app.name not in self.app.blueprints:
                self.app.register_blueprint(self.app_bp, url_prefix='/uit/dsp')

    def _setup_routes(self):
        @self.app_bp.route('', methods=['POST'])
        def preprocessing():
           return
    
    
    def run(self, port):
        self.app.run(port=port, debug=True)

