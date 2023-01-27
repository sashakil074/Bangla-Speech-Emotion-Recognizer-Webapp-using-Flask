from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# load array

#*** Backend operation
 
# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','wav'}
 
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templates', static_folder='staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model1 = load_model('C:/Users/shaki/OneDrive/Desktop/Speech Emotion/models/model.h5')
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
@app.route('/')
def index():
    return render_template('index_upload_and_display_audio.html')
"""def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title('Waveplot for audio with {} emotion'.format(emotion), size=20)
    librosa.display.waveshow(data,sr=sr)
    return plt.show()

def spectrogram(data, sr, emotion):
    x=librosa.stft(data)
    xdb=librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(12, 4))
    plt.title('spectrogram for audio with {} emotion'.format(emotion), size=20)
    librosa.display.specshow(xdb,sr=sr,x_axis='time',y_axis='hz')
    return plt.colorbar()"""
def noise(data):
  noise=np.random.normal(0,data.std(),data.size)
  augmented_data=data+noise*0.005          #noise factor=0.005
  return augmented_data
#We cannot pass raw audio files to our machine learning model as is, we would need to extract some features out of the audio data.

#We cannot pass raw audio files to our machine learning model as is, we would need to extract some features out of the audio data.

def extract_features2(data,sampling_rate):
  result=np.array([])

  #ZCR
  #The zero-crossing rate describes the rate at which a signal moves from positive to zero to negative or from negative to zero to positive.
  zcr=np.mean(librosa.feature.zero_crossing_rate(y=data).T,axis=0) #.T=Transpose
  result=np.hstack((result,zcr)) #stacking horizontally

  #croma_stft
  #Compute a chromagram from a waveform or power spectrogram.here we used energy (magnitude) spectrum
  stft=np.abs(librosa.stft(data))
  chroma_stft=np.mean(librosa.feature.chroma_stft(S=stft,sr=sampling_rate).T,axis=0)
  result=np.hstack((result,chroma_stft)) #stacking horizontally

  
  #spectral centroid
  cent = np.mean(librosa.feature.spectral_centroid(y=data, sr=sampling_rate))
  result=np.hstack((result,cent)) #stacking horizontally
  
  #spectral rolloff
  ro=np.mean(librosa.feature.spectral_rolloff(y=data, sr=sampling_rate))
  result = np.hstack((result, ro))

  """#Tempogram
  oenv=librosa.onset.onset_strength(y=data,sr=sampling_rate,hop_length=512)
  times=librosa.times_like(oenv,sr=sampling_rate,hop_length=512)
  tem = np.mean(librosa.feature.tempogram(onset_envelope=oenv,sr=sampling_rate,hop_length=512).T, axis=0)
  result = np.hstack((result, tem)) # stacking horizontally
  #chroma_cqt
  chroma_cq = np.mean(librosa.feature.chroma_cqt(y=data, sr=sampling_rate))
  result=np.hstack((result,chroma_cq)) #stacking horizontally
    
  #chroma_cens
  chroma_cens = np.mean(librosa.feature.chroma_cens(y=data, sr=sampling_rate))
  result=np.hstack((result,chroma_cens)) #stacking horizontally
  """
  
    
  #spectral contrast
  contrast = np.mean(librosa.feature.spectral_contrast(S= stft, sr=sampling_rate))
  result=np.hstack((result,contrast)) #stacking horizontally  

  #spectral flatness
  flatness = np.mean(librosa.feature.spectral_flatness(y=data))
  result=np.hstack((result,flatness)) #stacking horizontally  """
    
  #MFCC
  #Mel-frequency cepstral coefficients (MFCCs) form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale.
  mfcc=np.mean(librosa.feature.mfcc(y=data,sr=sampling_rate).T,axis=0)
  result=np.hstack((result,mfcc)) #stacking horizontally

  #Root Mean Square value(RMS)
  rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
  result = np.hstack((result, rms)) # stacking horizontally
 


  # MelSpectogram
  mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sampling_rate).T, axis=0)
  result = np.hstack((result, mel)) # stacking horizontally

  #return result

 

  return result

def get_features2(path):
  # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
  data, sampling_rate=librosa.load(path, duration=2.5, offset=0.6)
  #without augmentation
  res1=extract_features2(data,sampling_rate)
  result=np.array(res1)

  #with white noise
  noise_data=noise(data)
  res2=extract_features2(noise_data,sampling_rate)
  result=np.vstack((result, res2))   #stacking vertically
  """"
  #data with stretching and pitching
  stretch_noise=stretch(data,0.8)
  pitch_noise=pitch(stretch_noise,sampling_rate)
  res3=extract_features2(pitch_noise,sampling_rate)
  result=np.vstack((result, res3))   #stacking vertically

  #shifted noise
  shift_noise=shift(data)
  res4=extract_features2(shift_noise,sampling_rate)
  result=np.vstack((result, res4))   #stacking vertically"""

  return result
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    X=[]
    if request.method == 'POST':
        # Upload file flask
        uploaded_audio = request.files['uploaded-file']
        # Extracting uploaded data file name
        audio_filename = secure_filename(uploaded_audio.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_audio.save(os.path.join(app.config['UPLOAD_FOLDER'], audio_filename))
        # Storing uploaded file path in flask session
        session['uploaded_audio_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        path=os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file_path = session.get('uploaded_audio_file_path', None)

        ##Audio output display
        data, sampling_rate=librosa.load(path)
        emotion='unknown'
        #waveplot and spectrogram

        #waveplot=waveplot(data,sampling_rate,emotion)
        
        #IPython.display.Audio(path)
        #Audio(path)
        
        ##Feature Extraction
        X=[]
        Y=[]
        feature=get_features2(path)
        for element in feature:
             X.append(element)
             Y.append('unknown')
        Features2=pd.DataFrame(X)
        Features2['labels']=Y
        #Features2.to_csv('Subesco_features11_2.csv',index=False) #saving as csv
        Features1=pd.read_csv('Subesco_features11_1.csv')
        Xtrain = Features1.iloc[:, :-1].values

        Xtest = Features2.iloc[:, :-1].values
      
        #feature scaling
        sc = StandardScaler()
        Xtrain = sc.fit_transform(Xtrain)
        #test = sc.fit_transform(Xtest)
        Xtest = sc.transform(Xtest)
        
        Xtest = np.expand_dims(Xtest, axis=2)

        data = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        encoder = OneHotEncoder()
        Ytest = encoder.fit_transform(np.array(data).reshape(-1,1)).toarray()
        
        prediction = model1.predict(Xtest)
        
        pred_label = [np.argmax(i) for i in prediction]
        print(pred_label)
        y_pred3 = data[pred_label[0]]
        #y_pred3 = encoder.inverse_transform(prediction)

        return render_template('index_upload_and_display_audio.html', user_audio=audio_file_path, prediction_text='Emotion of Your Selected Audio is = {}'.format(np.char.capitalize(y_pred3)))
 
"""@app.route('/show_image')
def displayImage():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_img_file_path', None)
    # Display image in Flask application web page
    return render_template('index_upload_and_display_audio.html', user_image = img_file_path)"""
 
if __name__=='__main__':
    app.run(debug = True)