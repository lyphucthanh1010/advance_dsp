import os
from flask import Flask, request, make_response, Response, Blueprint,render_template,redirect, jsonify
from get_chromaprint_func import get_chromaprints, get_all_chromaprints
import json
from dotenv import load_dotenv
from flask_cors import CORS
import os
import librosa
import soundfile as sf
import shutil
import essentia.standard as es
import acoustid as ai

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
        
        # @self.app_bp.route('/preparedata', methods= ['POST'])
        # def prepare_data():
        #     return
        
        # @self.app_bp.route('/predict', methods = ['POST'])
        # def predict():
        #     return
        
        @self.app_bp.route('/get_chromaprints', methods = ['POST'])
        def get_chromaprints_api():
            data = request.get_json()
            if not data or 'audio_path' not in data:
                    return jsonify({'status': 'error', 'message':'Missing audio_path'}), 400
            audio_path = data['audio_path']
            is_save_res = data.get('is_save_res', True)
            try:
                segments = get_chromaprints(audio_path, is_save_res)
                return jsonify({'status': 'success', 'chromaprint_segments': segments})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app_bp.route('/get_all_chromaprints', methods= ['POST'])
        def get_all_chromaprints_api():
            data = request.get_json()
            if not data or 'file_paths' not in data:
                return jsonify({'status': 'error', 'message': 'Thiếu tham số file_paths'}), 400

            file_paths = data['file_paths']
            if not isinstance(file_paths, list):
                return jsonify({'status': 'error', 'message': 'file_paths phải là một danh sách các đường dẫn'}), 400

            try:
                results = get_all_chromaprints(file_paths)
                return jsonify({'status': 'success', 'results': results})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
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
            print("")
            return
        @self.app_bp.route('/move', methods = ['POST'])
        def move():
            data = request.get_json() or {}
            source_directory = data.get('source_directory', "Music")
            destination_directory = data.get('destination_directory', "music_dataset")
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            moved_files = []
            for root, dirs, files in os.walk(source_directory):
                for file in files:
                    ext= os.path.splitext(file)[1].lower()
                    if ext in self.audio_extensions:
                        source_file = os.path.join(root, file)
                        destination_file = os.path.join(destination_directory, file)
                        if os.path.exists(destination_file):
                            base, extension = os.path.splitext(file)
                            count = 1
                            new_file = f"{base}_{count}{extension}"
                            destination_file = os.path.join(destination_directory, new_file)
                            while os.path.exists(destination_file):
                                count += 1
                                new_file = f"{base}_{count}{extension}"
                                destination_file = os.path.join(destination_directory, new_file)

                        shutil.move(source_file, destination_file)
                        moved_files.append(destination_file)

            return jsonify({'status': 'success', 'moved_files': moved_files})
        
        @self.app_bp.route('/')
        def home():
            return render_template('index.html')

    def run(self):
        self.app.run(host='', port=os.getenv('FLASK_PORT'), debug=True)

if __name__ == "__main__":
    app_dsp = App()
    app_dsp.run()