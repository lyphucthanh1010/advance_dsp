import os
from flask import Flask, request, make_response, Response, Blueprint,render_template,redirect, jsonify
from get_chromaprint_func import get_chromaprints, get_all_chromaprints
from segment_song_func import process_audio_file
import json
from dotenv import load_dotenv
from flask_cors import CORS
import os
import librosa
import soundfile as sf
import shutil
import multiprocessing
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
        
        @self.app_bp.route('/preparedata', methods= ['POST'])
        def prepare_data():
            return
        
        @self.app_bp.route('/predict', methods = ['POST'])
        def predict():
            return
        
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

        @self.app_bp.route('/get_all_chromaprints', methods=['POST'])
        def get_all_chromaprints_api():
            data = request.get_json()
            if not data or 'source_directory' not in data or 'destination_directory' not in data:
                return jsonify({'status': 'error', 'message': 'Thiếu tham số source_directory hoặc destination_directory'}), 400

            source_directory = data['source_directory']
            destination_directory = data['destination_directory']

            if not os.path.exists(source_directory):
                return jsonify({'status': 'error', 'message': f"Thư mục nguồn '{source_directory}' không tồn tại."}), 400

            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            args_list = []
            for root, dirs, files in os.walk(source_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Tính đường dẫn tương đối từ thư mục nguồn
                    relative_path = os.path.relpath(file_path, source_directory)
                    # Tạo đường dẫn file đích, thay thế đuôi (.wav, .mp3, ...) bằng .json
                    dest_file_path = os.path.splitext(os.path.join(destination_directory, relative_path))[0] + ".json"
                    # Đảm bảo thư mục chứa file đích tồn tại
                    dest_dir = os.path.dirname(dest_file_path)
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir, exist_ok=True)
                    args_list.append((file_path, dest_file_path))

            processed_files = []
            with multiprocessing.Pool() as pool:
                results = pool.map(get_chromaprints, args_list)
                processed_files = [r for r in results if r is not None]

            return jsonify({
                'status': 'success',
                'total_processed': len(processed_files),
                'processed_files': processed_files
            })
        
        @self.app_bp.route('/convert', methods = ['POST'])
        def convert2wav():
            data = request.get_json()
            if not data or 'source_directory' not in data or 'destination_directory' not in data:
                return jsonify({
                    'status': 'error',
                    'message': 'Vui lòng cung cấp "source_directory" và "destination_directory".'
                }), 400

            source_directory = data['source_directory']
            destination_directory = data['destination_directory']

            if not os.path.exists(source_directory):
                return jsonify({
                    'status': 'error',
                    'message': f"Thư mục nguồn '{source_directory}' không tồn tại."
                }), 400

            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            converted_files = []
            for root, dirs, files in os.walk(source_directory):
                for file in files:
                    if file.lower().endswith(".mp3"):
                        source_file_path = os.path.join(root, file)
                        base_name = os.path.splitext(file)[0]
                        destination_file_path = os.path.join(destination_directory, base_name + ".wav")
                        try:
                            y, sr = librosa.load(source_file_path, sr=None)
                            sf.write(destination_file_path, y, sr)
                            converted_files.append(destination_file_path)
                            print(f"Chuyển đổi thành công : {base_name}")
                        except Exception as e:
                            print(f"Lỗi khi chuyển đổi {source_file_path}: {e}")

            return jsonify({
                'status': 'success',
                'converted_files': converted_files
            })
        
        @self.app_bp.route('/segment', methods = ['POST'])
        def segment_song():
            audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
            data = request.get_json()
            if not data or 'source_directory' not in data or 'destination_directory' not in data:
                return jsonify({
                    'status': 'error',
                    'message': 'Vui lòng cung cấp "source_directory" và "destination_directory".'
                }), 400

            source_directory = data['source_directory']
            destination_directory = data['destination_directory']

            if not os.path.exists(source_directory):
                return jsonify({
                    'status': 'error',
                    'message': f"Thư mục nguồn '{source_directory}' không tồn tại."
                }), 400

            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            segment_duration = 10  
            step_duration = 2
            audio_files = []
            for root, dirs, files in os.walk(source_directory):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in audio_extensions:
                        audio_files.append(os.path.join(root, file))
                        print(file)

            pool_args = [(audio_file, destination_directory, segment_duration, step_duration) for audio_file in audio_files]

            all_segments = []
            with multiprocessing.Pool() as pool:
                results = pool.map(process_audio_file, pool_args)
                for segments in results:
                    all_segments.extend(segments)

            return jsonify({
                'status': 'success',
                'total_segments': len(all_segments),
                'segments': all_segments
            })

            
        
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