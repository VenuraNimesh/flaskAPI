from flask import Flask, request, redirect,jsonify,url_for, Response,send_from_directory, send_file
from werkzeug.datastructures import FileStorage
import os
import librosa
import IPython.display as ipd
import librosa.display
import numpy as np
import audioread
import scipy
import wave
import io
from denoisedPart import denoise_audio
from denoise import denoiseFil
import matplotlib.pyplot as plt
from wave import Wave_write
from werkzeug import secure_filename
windowLength = 256
ffTLength = windowLength
overlap      = round(0.25 * windowLength)
fs           = 22050
numSegments  = 8
numFeatures  = ffTLength//2 + 1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route("/predict", methods=["GET","POST"])


def predict():
    data = {"success": False}
    if request.method == 'POST':
        soundFile = request.files['file']
        
        if soundFile:
            filename = secure_filename(soundFile.filename)
            soundFile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
            noiseFilePath = app.config['UPLOAD_FOLDER'] + filename
            
            denoise_audio(noiseFilePath)
            #denoiseFil(noiseFilePath)

            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            filename = 'denoised.wav'
            noiseFilePath1 = app.config['UPLOAD_FOLDER'] + filename
            print(noiseFilePath1)
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":   
    app.run()
