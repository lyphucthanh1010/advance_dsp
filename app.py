import os
from flask import Flask, request, make_response, Response, Blueprint,render_template,redirect, jsonify
import json
from dotenv import load_dotenv
from flask_cors import CORS
import os
import librosa
import soundfile as sf

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
        
        # @self.app_bp.route('/check', methods= ['POST'])
        # def check():
        #     return 
        
        @self.app_bp.route('/convert', methods = ['POST'])
        def convert2wav(input_path, output_folder):
            if not os.path.isfile( input_path):
                raise FileNotFoundError(f"Tệp đầu vào không tồn tại: {input_path}")

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            base_filename = os.path.splitext(os.path.basename(input_path))[0]
            wav_filename = base_filename + ".wav"
            wav_path = os.path.join(output_folder, wav_filename)

            try:
                y, sr = librosa.load(input_path, sr=None, mono=False)

                if y.ndim == 1:
                    data = y
                else:
                    data = y.T
                sf.write(wav_path, data, sr, subtype='PCM_16') 

                return wav_path

            except Exception as e:
                print(f"Lỗi khi chuyển đổi tệp {input_path}: {e}")
                raise

        
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
        self.app.run(host='', port=os.getenv('FLASK_PORT'), debug=True)

if __name__ == "__main__":
    app_dsp = App()
    app_dsp.run()

Dummy commit cho ngày 2024-01-01
Dummy commit cho ngày 2024-01-02
Dummy commit cho ngày 2024-01-03
Dummy commit cho ngày 2024-01-04
Dummy commit cho ngày 2024-01-05
Dummy commit cho ngày 2024-01-06
Dummy commit cho ngày 2024-01-07
Dummy commit cho ngày 2024-01-08
Dummy commit cho ngày 2024-01-09
Dummy commit cho ngày 2024-01-10
Dummy commit cho ngày 2024-01-11
Dummy commit cho ngày 2024-01-12
Dummy commit cho ngày 2024-01-13
Dummy commit cho ngày 2024-01-14
Dummy commit cho ngày 2024-01-15
Dummy commit cho ngày 2024-01-16
