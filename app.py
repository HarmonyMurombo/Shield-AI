import argparse
import io
from PIL import Image
from datetime import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from twilio.rest import Client
from flask_mail import Mail,Message
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
#Twilio credentials
# Get Twilio credentials from the environment variables
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER') 

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

#Function to send SMS via Twilio
def send_sms_via_twilio(phone_number, message):
    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        print(f"SMS sent: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")
        
#Initialising the application
app = Flask(__name__)

#Configure the database URI and secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'secretkey'

#Initialize the database and bcrypt
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define your table models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    
with app.app_context():
        db.create_all()  # Create all table
    
class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="AOqc0EA9cM5fBdZmfjVA"
)

# The default route (homepage)
@app.route("/")
def home():
    return render_template("home.html")

#The route for starting and saving the images
@app.route("/index")
def return_imge():
    return render_template("index.html")

#Login route 
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('return_imge'))  # Updated here
    return render_template('login.html', form=form)

#The index page 
@app.route('/index', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('index.html')


#The route for logging
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

#Route for registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

#The default route 
@app.route("/predict", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)
                                               
            file_extension = f.filename.rsplit('.', 1)[1].lower() 
            
            if file_extension == 'jpg':
                img = cv2.imread(filepath)

                # Perform the detection
                model = YOLO('yolo11.pt')
                detections =  model(img, save=True) 
              
                # Send SMS if a weapon is detected
                if len(detections) > 0:  # If any object detected
                    send_sms_via_twilio("+263780517601", "Attention!!, a weapon has been detected be on high alert!!")
                return display(f.filename)
           
            elif file_extension == 'mp4': 
                video_path = filepath  # replace with your video path
                cap = cv2.VideoCapture(video_path)

                # get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                
                # initialize the YOLOv11 model here
                model = YOLO('yolo11.pt')
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break                                                      

                    # do YOLOv11 detection on the frame here
                    #model = YOLO('yolov11c.pt')
                    results = model(frame, save=True)  #working
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)
                    
                    # write the frame to the output video
                    out.write(res_plotted)
                    
                    # Send SMS if a weapon is detected
                    if len(results) > 0:
                        send_sms_via_twilio("+263780517601", "Attention!!, a weapon has been detected be on high alert!!")
                    if cv2.waitKey(1) == ord('q'):
                        break

                return video_feed()      
            
#Function for the current location of the device by using its ip address       
def get_current_location():
    try:
        # Fetch location data using ip-api
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        if data["status"] == "success":
            # Return latitude and longitude
            return data["lat"], data["lon"]
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching location: {e}")
        return None, None   
     
#The route for the notifications of sending and receiving alerts to the alerts page
@app.route('/notifications')
def notifications():
    # Folder containing detection results
    folder_path = 'runs/detect'
    
    # Find the latest subfolder in the detection results directory
    subfolders = [
        f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))
    ]
    
    if not subfolders:
        # If no detections exist, return an empty page
        return render_template('notifications.html', alerts=[])
    
    latest_subfolder = max(
        subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x))
    )
    
    # Build the full path to the latest detection images
    latest_folder_path = os.path.join(folder_path, latest_subfolder)
    images = [
        os.path.join(latest_folder_path, img)
        for img in os.listdir(latest_folder_path)
        if img.endswith(('.jpg', '.png'))
    ]
    
    # Get the current location
    latitude, longitude = get_current_location()
    map_link = (
        f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude
        else "https://www.google.com/maps"  # Fallback if location is unavailable
    )
    
    # Prepare alert data with timestamps and locations
    alerts = []
    for img_path in images:
        alert = {
            "image_url": img_path.replace('\\', '/'),  # Ensure URL compatibility
            "time": datetime.now().strftime("%Y-%m-%d %I:%M %p"),  # Current time
            "map_link": map_link
        }
        alerts.append(alert)
    
    # Render the notifications page with the alerts
    return render_template('notifications.html', alerts=alerts)


# #The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder    
    print("printing directory: ",directory) 
    files = os.listdir(directory)
    latest_file = files[0]
    
    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

    else:
        return "Invalid file format"
        
def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
      
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")

    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
        
#Function to start the cctv camera in realtime and detect guns and knives
@app.route("/cctv_feed")
def cctv_feed():
    cap = cv2.VideoCapture(0)
    
    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            img = Image.open(io.BytesIO(frame))
            model = YOLO('yolo11.pt')
            results = model(img, save=True)
            res_plotted = results[0].plot()
            
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            
              # Send SMS if a weapon is detected
            if len(results) > 0:
                send_sms_via_twilio("+263780517601", "Attention!!, a weapon has been detected be on high alert!!")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for concealed detection image upload
@app.route('/concealed_detection', methods=['GET', 'POST'])
def concealed_detection():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads')

            # Create uploads folder if it doesn't exist
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Save the uploaded file
            filepath = os.path.join(upload_folder, secure_filename(f.filename))
            f.save(filepath)

            # Perform inference
            try:
                result = CLIENT.infer(filepath, model_id="shield-ai/2")
                print("Inference result:", result)

                # Open the image using OpenCV
                img = cv2.imread(filepath)

                # Draw bounding boxes based on predictions
                for pred in result['predictions']:
                    x = int(pred['x'] - pred['width'] / 2)
                    y = int(pred['y'] - pred['height'] / 2)
                    w = int(pred['width'])
                    h = int(pred['height'])
                    confidence = pred['confidence']
                    class_label = pred['class']

                    # Draw rectangle (bounding box)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display the class and confidence score
                    label = f"{class_label}: {confidence:.2f}"
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the annotated image
                output_path = os.path.join(upload_folder, 'result.jpg')
                cv2.imwrite(output_path, img)

                # Send the image back to the client
                return send_file(output_path, mimetype='image/jpeg')

            except Exception as e:
                print(f"Error during inference: {e}")
                return "An error occurred during detection", 500

    # Render the upload form if not a POST request
    return render_template('index.html')


# Route to generate real-time thermal camera feed with detection
def generate_thermal_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to grayscale and apply the thermal effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_effect = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # Save the frame temporarily for inference (as CLIENT expects a file path)
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, thermal_effect)

        # Perform inference using your model
        try:
            result = CLIENT.infer(temp_path, model_id="shield-ai/2")
            print("Inference result:", result)

            # Draw bounding boxes based on predictions
            for pred in result['predictions']:
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])
                confidence = pred['confidence']
                class_label = pred['class']

                # Draw rectangle (bounding box)
                cv2.rectangle(thermal_effect, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the class and confidence score
                label = f"{class_label}: {confidence:.2f}"
                cv2.putText(thermal_effect, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error during inference: {e}")
        
        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', thermal_effect)
        if not ret:
            break

        # Yield the frame for streaming
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#Start CCTV
@app.route('/start_thermal_camera')
def start_thermal_camera():
    return Response(generate_thermal_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

#Route for the contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

#Code for atlas assistant
@app.route('/atlas')
def atlas():
    return render_template('atlas.html')

# The default route (homepage)
@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))