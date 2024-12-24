import os
from flask import Flask, request, make_response, Response, Blueprint,render_template,redirect
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
        @self.app.route('/')
        def redirect_to_dsp():
            return redirect('/uit/dsp/')
        
        @self.app_bp.route('/preprocess', methods=['POST'])
        def preprocessing():
            return
        
        @self.app_bp.route('/preparedata', methods= ['POST'])
        def prepare_data():
            return
        
        @self.app_bp.route('/predict', methods = ['POST'])
        def predict():
            return
        
        @self.app_bp.route('/convert', methods = ['POST'])
        def convert2wav():
            return
        
        @self.app_bp.route('/segment', methods = ['POST'])
        def segment_song():
            return
        
        @self.app_bp.route('/augment', methods = ['POST'])
        def augment():
            return
        
        @self.app_bp.route('/train', methods = ['POST'])
        def train():
            return
        
        @self.app_bp.route('/validate', methods = ['POST'])
        def validate():
            return
        
        @self.app_bp.route('/predict_youtube', methods = ['POST'])
        def predict_youtube():
            return

        @self.app_bp.route('/')
        def home():
            return render_template('index.html')

    def run(self):
        self.app.run(host='localhost', port=os.getenv('FLASK_PORT'), debug=True)

if __name__ == "__main__":
    app_dsp = App()
    app_dsp.run()

