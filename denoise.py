import numpy as np
import scipy as sp
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy import signal
import os

def denoiseFil(filename):
    (Frequency, array) = read(os.path.join(filename))
    
    FourierTransformation = sp.fft(array)

    scale = sp.linspace(0, Frequency, len(array))

    GuassianNoise = np.random.rand(len(FourierTransformation))

    NewSound = GuassianNoise + array

    b,a = signal.butter(4, 5000/(Frequency/2), btype='highpass')

    filteredSignal = signal.lfilter(b,a,NewSound)
    
    c,d = signal.butter(5,380/(Frequency/2), btype='lowpass')
    newFilteredSignal = signal.lfilter(c,d,filteredSignal) 

    write("uploads/denoised.wav", Frequency, newFilteredSignal)