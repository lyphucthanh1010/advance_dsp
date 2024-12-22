import essentia
import essentia.standard as es

def extract_music_features(audio_path):
    extractor = es.MusicExtractor()
    features = extractor(audio_path)
    return features

if __name__ == "__main__":
    audio_file = "TraiHoVubeat.mp3"
    features = extract_music_features(audio_file)
    print(features)
