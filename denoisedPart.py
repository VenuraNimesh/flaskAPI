import os
import librosa
import IPython.display as ipd
import librosa.display
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
windowLength = 256
ffTLength = windowLength
overlap      = round(0.25 * windowLength)
fs           = 16000
numSegments  = 8
numFeatures  = ffTLength//2 + 1

model = load_model('model/denoiser_cnn_colab120X536.h5')

def denoise_audio(filename):
    noiseAudio, sr = read_audio(os.path.join(filename), sample_rate=fs)
    print("Min:", np.min(noiseAudio),"Max:",np.max(noiseAudio))
    ipd.Audio(data=noiseAudio, rate=sr)
    
    noiseAudioFeatureExtractor = FeatureExtractor(noiseAudio, windowLength=windowLength, overlap=overlap, sample_rate=sr)
    noise_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()
    
    noisyPhase = np.angle(noise_stft_features)
    print(noisyPhase.shape)
    noise_stft_features = np.abs(noise_stft_features)
    
    mean = np.mean(noise_stft_features)
    std = np.std(noise_stft_features)
    noise_stft_features = (noise_stft_features - mean) / std
    
    predictors = prepare_input_features(noise_stft_features)
    predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
    predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
    print('predictors.shape:', predictors.shape)
    
    STFTFullyConvolutional = model.predict(predictors)
    print(STFTFullyConvolutional.shape)
   
    denoisedAudioFullyConvolutional = revert_features_to_audio(STFTFullyConvolutional, noisyPhase, noiseAudioFeatureExtractor, mean, std)
    print("Min:", np.min(denoisedAudioFullyConvolutional),"Max:",np.max(denoisedAudioFullyConvolutional))  
    denoised = librosa.output.write_wav('uploads/denoised.wav', denoisedAudioFullyConvolutional, fs)
    return denoised

def read_audio(filepath, sample_rate, normalize=True):
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize:
      div_fac = 1 / np.max(np.abs(audio)) / 3.0
      audio = audio * div_fac
    return audio, sr

def play(audio, sample_rate):
    ipd.display(ipd.Audio(data=audio, rate=sample_rate))
        
class FeatureExtractor:
    def __init__(self, audio, *, windowLength, overlap, sample_rate):
        self.audio = audio
        self.ffT_length = windowLength
        self.window_length = windowLength
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.window = scipy.signal.hamming(self.window_length, sym=False)

    def get_stft_spectrogram(self):
        return librosa.stft(self.audio, n_fft=self.ffT_length, win_length=self.window_length, hop_length=self.overlap,
                            window=self.window, center=True)

    def get_audio_from_stft_spectrogram(self, stft_features):
        return librosa.istft(stft_features, win_length=self.window_length, hop_length=self.overlap, window=self.window, center=True)

    def get_mel_spectrogram(self):
        return librosa.feature.melspectrogram(self.audio, sr=self.sample_rate, power=2.0, pad_mode='reflect', n_fft=self.ffT_length, hop_length=self.overlap, center=True)

    def get_audio_from_mel_spectrogram(self, M):
        return librosa.feature.inverse.mel_to_audio(M, sr=self.sample_rate, n_fft=self.ffT_length, hop_length=self.overlap, win_length=self.window_length, window=self.window, center=True, pad_mode='reflect', power=2.0, n_iter=32, length=None)

def prepare_input_features(stft_features):
    noisySTFT = np.concatenate([stft_features[:,0:numSegments-1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments , noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:,:,index] = noisySTFT[:,index:index + numSegments]
    return stftSegments

def revert_features_to_audio(features, phase, noiseAudioFeatureExtractor, cleanMean=None, cleanStd=None):
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)

    features = features * np.exp(1j * phase) 
    
    features = np.transpose(features, (1, 0))
    return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)
