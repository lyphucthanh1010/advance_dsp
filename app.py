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
Dummy commit cho ngày 2024-01-17
Dummy commit cho ngày 2024-01-18
Dummy commit cho ngày 2024-01-19
Dummy commit cho ngày 2024-01-20
Dummy commit cho ngày 2024-01-21
Dummy commit cho ngày 2024-01-22
Dummy commit cho ngày 2024-01-23
Dummy commit cho ngày 2024-01-24
Dummy commit cho ngày 2024-01-25
Dummy commit cho ngày 2024-01-26
Dummy commit cho ngày 2024-01-27
Dummy commit cho ngày 2024-01-28
Dummy commit cho ngày 2024-01-29
Dummy commit cho ngày 2024-01-30
Dummy commit cho ngày 2024-01-31
Dummy commit cho ngày 2024-02-01
Dummy commit cho ngày 2024-02-02
Dummy commit cho ngày 2024-02-03
Dummy commit cho ngày 2024-02-04
Dummy commit cho ngày 2024-02-05
Dummy commit cho ngày 2024-02-06
Dummy commit cho ngày 2024-02-07
Dummy commit cho ngày 2024-02-08
Dummy commit cho ngày 2024-02-09
Dummy commit cho ngày 2024-02-10
Dummy commit cho ngày 2024-02-11
Dummy commit cho ngày 2024-02-12
Dummy commit cho ngày 2024-02-13
Dummy commit cho ngày 2024-02-14
Dummy commit cho ngày 2024-02-15
Dummy commit cho ngày 2024-02-16
Dummy commit cho ngày 2024-02-17
Dummy commit cho ngày 2024-02-18
Dummy commit cho ngày 2024-02-19
Dummy commit cho ngày 2024-02-20
Dummy commit cho ngày 2024-02-21
Dummy commit cho ngày 2024-02-22
Dummy commit cho ngày 2024-02-23
Dummy commit cho ngày 2024-02-24
Dummy commit cho ngày 2024-02-25
Dummy commit cho ngày 2024-02-26
Dummy commit cho ngày 2024-02-27
Dummy commit cho ngày 2024-02-28
Dummy commit cho ngày 2024-02-29
Dummy commit cho ngày 2024-03-01
Dummy commit cho ngày 2024-03-02
Dummy commit cho ngày 2024-03-03
Dummy commit cho ngày 2024-03-04
Dummy commit cho ngày 2024-03-05
Dummy commit cho ngày 2024-03-06
Dummy commit cho ngày 2024-03-07
Dummy commit cho ngày 2024-03-08
Dummy commit cho ngày 2024-03-09
Dummy commit cho ngày 2024-03-10
Dummy commit cho ngày 2024-03-11
Dummy commit cho ngày 2024-03-12
Dummy commit cho ngày 2024-03-13
Dummy commit cho ngày 2024-03-14
Dummy commit cho ngày 2024-03-15
Dummy commit cho ngày 2024-03-16
Dummy commit cho ngày 2024-03-17
Dummy commit cho ngày 2024-03-18
Dummy commit cho ngày 2024-03-19
Dummy commit cho ngày 2024-03-20
Dummy commit cho ngày 2024-03-21
Dummy commit cho ngày 2024-03-22
Dummy commit cho ngày 2024-03-23
Dummy commit cho ngày 2024-03-24
Dummy commit cho ngày 2024-03-25
Dummy commit cho ngày 2024-03-26
Dummy commit cho ngày 2024-03-27
Dummy commit cho ngày 2024-03-28
Dummy commit cho ngày 2024-03-29
Dummy commit cho ngày 2024-03-30
Dummy commit cho ngày 2024-03-31
Dummy commit cho ngày 2024-04-01
Dummy commit cho ngày 2024-04-02
Dummy commit cho ngày 2024-04-03
Dummy commit cho ngày 2024-04-04
Dummy commit cho ngày 2024-04-05
Dummy commit cho ngày 2024-04-06
Dummy commit cho ngày 2024-04-07
Dummy commit cho ngày 2024-04-08
Dummy commit cho ngày 2024-04-09
Dummy commit cho ngày 2024-04-10
Dummy commit cho ngày 2024-04-11
Dummy commit cho ngày 2024-04-12
Dummy commit cho ngày 2024-04-13
Dummy commit cho ngày 2024-04-14
Dummy commit cho ngày 2024-04-15
Dummy commit cho ngày 2024-04-16
Dummy commit cho ngày 2024-04-17
Dummy commit cho ngày 2024-04-18
Dummy commit cho ngày 2024-04-19
Dummy commit cho ngày 2024-04-20
Dummy commit cho ngày 2024-04-21
Dummy commit cho ngày 2024-04-22
Dummy commit cho ngày 2024-04-23
Dummy commit cho ngày 2024-04-24
Dummy commit cho ngày 2024-04-25
Dummy commit cho ngày 2024-04-26
Dummy commit cho ngày 2024-04-27
Dummy commit cho ngày 2024-04-28
Dummy commit cho ngày 2024-04-29
Dummy commit cho ngày 2024-04-30
Dummy commit cho ngày 2024-05-01
Dummy commit cho ngày 2024-05-02
Dummy commit cho ngày 2024-05-03
Dummy commit cho ngày 2024-05-04
Dummy commit cho ngày 2024-05-05
Dummy commit cho ngày 2024-05-06
Dummy commit cho ngày 2024-05-07
Dummy commit cho ngày 2024-05-08
Dummy commit cho ngày 2024-05-09
Dummy commit cho ngày 2024-05-10
Dummy commit cho ngày 2024-05-11
Dummy commit cho ngày 2024-05-12
Dummy commit cho ngày 2024-05-13
Dummy commit cho ngày 2024-05-14
Dummy commit cho ngày 2024-05-15
Dummy commit cho ngày 2024-05-16
Dummy commit cho ngày 2024-05-17
Dummy commit cho ngày 2024-05-18
Dummy commit cho ngày 2024-05-19
Dummy commit cho ngày 2024-05-20
Dummy commit cho ngày 2024-05-21
Dummy commit cho ngày 2024-05-22
Dummy commit cho ngày 2024-05-23
Dummy commit cho ngày 2024-05-24
Dummy commit cho ngày 2024-05-25
Dummy commit cho ngày 2024-05-26
Dummy commit cho ngày 2024-05-27
Dummy commit cho ngày 2024-05-28
Dummy commit cho ngày 2024-05-29
Dummy commit cho ngày 2024-05-30
Dummy commit cho ngày 2024-05-31
Dummy commit cho ngày 2024-06-01
Dummy commit cho ngày 2024-06-02
Dummy commit cho ngày 2024-06-03
Dummy commit cho ngày 2024-06-04
Dummy commit cho ngày 2024-06-05
Dummy commit cho ngày 2024-06-06
Dummy commit cho ngày 2024-06-07
Dummy commit cho ngày 2024-06-08
Dummy commit cho ngày 2024-06-09
Dummy commit cho ngày 2024-06-10
Dummy commit cho ngày 2024-06-11
Dummy commit cho ngày 2024-06-12
Dummy commit cho ngày 2024-06-13
Dummy commit cho ngày 2024-06-14
Dummy commit cho ngày 2024-06-15
Dummy commit cho ngày 2024-06-16
Dummy commit cho ngày 2024-06-17
Dummy commit cho ngày 2024-06-18
Dummy commit cho ngày 2024-06-19
Dummy commit cho ngày 2024-06-20
Dummy commit cho ngày 2024-06-21
Dummy commit cho ngày 2024-06-22
Dummy commit cho ngày 2024-06-23
Dummy commit cho ngày 2024-06-24
Dummy commit cho ngày 2024-06-25
Dummy commit cho ngày 2024-06-26
Dummy commit cho ngày 2024-06-27
Dummy commit cho ngày 2024-06-28
Dummy commit cho ngày 2024-06-29
Dummy commit cho ngày 2024-06-30
Dummy commit cho ngày 2024-07-01
Dummy commit cho ngày 2024-07-02
Dummy commit cho ngày 2024-07-03
Dummy commit cho ngày 2024-07-04
Dummy commit cho ngày 2024-07-05
Dummy commit cho ngày 2024-07-06
Dummy commit cho ngày 2024-07-07
Dummy commit cho ngày 2024-07-08
Dummy commit cho ngày 2024-07-09
Dummy commit cho ngày 2024-07-10
Dummy commit cho ngày 2024-07-11
Dummy commit cho ngày 2024-07-12
Dummy commit cho ngày 2024-07-13
Dummy commit cho ngày 2024-07-14
Dummy commit cho ngày 2024-07-15
Dummy commit cho ngày 2024-07-16
Dummy commit cho ngày 2024-07-17
Dummy commit cho ngày 2024-07-18
Dummy commit cho ngày 2024-07-19
Dummy commit cho ngày 2024-07-20
Dummy commit cho ngày 2024-07-21
Dummy commit cho ngày 2024-07-22
Dummy commit cho ngày 2024-07-23
Dummy commit cho ngày 2024-07-24
Dummy commit cho ngày 2024-07-25
Dummy commit cho ngày 2024-07-26
Dummy commit cho ngày 2024-07-27
Dummy commit cho ngày 2024-07-28
Dummy commit cho ngày 2024-07-29
Dummy commit cho ngày 2024-07-30
Dummy commit cho ngày 2024-07-31
Dummy commit cho ngày 2024-08-01
Dummy commit cho ngày 2024-08-02
Dummy commit cho ngày 2024-08-03
Dummy commit cho ngày 2024-08-04
Dummy commit cho ngày 2024-08-05
Dummy commit cho ngày 2024-08-06
Dummy commit cho ngày 2024-08-07
Dummy commit cho ngày 2024-08-08
Dummy commit cho ngày 2024-08-09
Dummy commit cho ngày 2024-08-10
Dummy commit cho ngày 2024-08-11
Dummy commit cho ngày 2024-08-12
Dummy commit cho ngày 2024-08-13
Dummy commit cho ngày 2024-08-14
Dummy commit cho ngày 2024-08-15
