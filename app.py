from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
# from pushbullet import PushBullet
from werkzeug.utils import secure_filename
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

brain_model= load_model('braintumor.h5')
covid_model= load_model('Covid.h5')

UPLOAD_FOLDER='static/uploads'
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

app.config['TEMPLATES_AUTO_RELOAD']=True

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')



@app.route('/resultb' , methods=['POST'])
def resultb():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file :
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img= cv2.imread('static/uploads/'+filename)
            img= Image.fromarray(img)
            img=img.resize((64,64))
            img=np.array(img)
            input=np.expand_dims(img, axis=0)
            pred= brain_model.predict(input)

            return render_template('resultb.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)
        
@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file :
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path='static/uploads/'+filename
            img=load_img(path, target_size=(256,256,3))
            img= img_to_array(img)/255
            img=np.array([img])
            pred = covid_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

    



if __name__ == '__main__':
    app.run(debug=True)

