import librosa
import soundfile as sf

y, sr = librosa.load("test.wav", sr=None, duration=10)
sf.write("short.wav", y, sr)