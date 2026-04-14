import librosa
import soundfile as sf

y, sr = librosa.load("test.wav", sr=None, duration=60)
sf.write("minute_test.wav", y, sr)