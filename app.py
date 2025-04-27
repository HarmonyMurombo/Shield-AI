import argparse
import io
from PIL import Image
from datetime import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, jsonify, send_from_directory, flash, abort
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
from flask_mail import Mail, Message
from dotenv import load_dotenv
import tempfile
from flask import session


# Load environment variables from .env file
load_dotenv()

# Twilio credentials
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER') 
TWILIO_WHATSAPP_NUMBER= os.getenv('TWILIO_WHATSAPP_NUMBER') 
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Function to send SMS via Twilio
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
        

# Function to send alert via whatsapp
def send_whatsapp_via_twilio(phone_number, message, media_url=None):
    try:
        from_whatsapp = 'whatsapp:+14155238886'  # Twilio's sandbox number
        to_whatsapp = 'whatsapp:' + phone_number

        # Log the media URL for debugging purposes
        if media_url:
            print(f"Media URL to be sent: {media_url}")

        msg = client.messages.create(
            body=message,
            from_=from_whatsapp,
            to=to_whatsapp,
            media_url=[media_url] if media_url else None
        )
        print(f"WhatsApp message sent: {msg.sid}")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

# Initialising the application
app = Flask(__name__)

# Configure the database URI and secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'secretkey'

# Initialize the database and bcrypt
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
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
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

# The route for starting and saving the images
@app.route("/index")
def return_imge():
    return render_template("index.html")

# Login route 
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('return_imge'))
        else:
            flash('Invalid username or password', 'error')  # Flash error message
    return render_template('login.html', form=form)

# The index page 
@app.route('/index', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('index.html')

# The route for logging out
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Route for registration
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

# Create a dedicated temp directory
TEMP_IMAGE_DIR = os.path.join(os.getcwd(), 'temp_images')
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

@app.route('/temp_images/<filename>')
def temp_images(filename):
    return send_from_directory(TEMP_IMAGE_DIR, filename)

# The default route for image prediction
@app.route("/predict", methods=["GET", "POST"])
def predict_img():
    def get_latest_detection_image():
        # Existing detection folder logic
        detection_folders = glob.glob('runs/detect/*/')
        if not detection_folders:
            return None
        latest_folder = max(detection_folders, key=os.path.getctime)
        images = glob.glob(os.path.join(latest_folder, '*.jpg'))
        return images[0] if images else None

    def send_weapon_alert(class_name, confidence, source_type="image"):
        # Existing alert logic
        try:
            # Get location data
            lat = session.get('user_lat')
            lon = session.get('user_lon')
            location_msg = f"Location: https://maps.google.com/?q={lat},{lon}" if lat and lon else "Location data unavailable"

            # Construct message
            alert_message = (
                f"🚨 WEAPON DETECTION ALERT 🚨\n"
                f"Type: {source_type.upper()}\n"
                f"Class: {class_name.upper()}\n"
                f"Confidence: {confidence*100:.2f}%\n"
                f"{location_msg}"
            )

            # Get detection image
            detected_img_path = get_latest_detection_image()
            
            if detected_img_path:
                # Create temporary file with unique name
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    temp_path = temp_file.name
                    shutil.copy(detected_img_path, temp_path)
                    
                # Use ngrok URL (replace with your actual ngrok URL)
                base_url = "https://killdeer-probable-mouse.ngrok-free.app"  # Update this
                media_url = f"{base_url}/temp_images/{os.path.basename(temp_path)}"
                
                # Send WhatsApp with image
                send_whatsapp_via_twilio("+263780517601", alert_message, media_url)
            
            # Also send SMS
            send_sms_via_twilio("+263780517601", alert_message)

        except Exception as e:
            print(f"Error sending alert: {e}")
            
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            filepath = os.path.join(upload_dir, f.filename)
            f.save(filepath)
            
            file_extension = f.filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'jpg':
                # Process image
                model = YOLO('yolo12.pt')
                results = model(filepath, save=True)
                
                # Check detections
                weapon_detected = False
                for result in results:
                    for box in result.boxes:
                        cls_idx = int(box.cls.item())
                        class_name = model.names[cls_idx]
                        confidence = box.conf.item()
                        if class_name in ['gun', 'knife']:
                            weapon_detected = True
                            send_weapon_alert(class_name, confidence)

                if not weapon_detected:
                    print("No weapons detected")
                
                return display(f.filename)
           
            elif file_extension == 'mp4':
                # Video processing
                video_path = filepath
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                
                model = YOLO('yolo12.pt')
                last_alert_time = 0
                alert_cooldown = 30  # seconds

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model(frame, save=True)
                    res_plotted = results[0].plot()
                    out.write(res_plotted)
                    
                    # Check for weapons in current frame
                    current_time = time.time()
                    for result in results:
                        for box in result.boxes:
                            cls_idx = int(box.cls.item())
                            class_name = model.names[cls_idx]
                            confidence = box.conf.item()
                            if class_name in ['gun', 'knife'] and (current_time - last_alert_time) > alert_cooldown:
                                send_weapon_alert(class_name, confidence, "video")
                                last_alert_time = current_time

                    if cv2.waitKey(1) == ord('q'):
                        break

                cap.release()
                out.release()
                return video_feed()

    return redirect(request.url)

# Function for fetching the current location using IP
def get_current_location():
    try:
        # First try to get from session (client-side geolocation)
        client_lat = session.get('user_lat')
        client_lon = session.get('user_lon')
        
        if client_lat and client_lon:
            print("Using client-side location")
            return float(client_lat), float(client_lon)
        
        # Fallback to IP-based location
        print("Falling back to IP geolocation")
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        if data["status"] == "success":
            return data["lat"], data["lon"]
        return None, None
    except Exception as e:
        print(f"Error fetching location: {e}")
        return None, None
    
# Route for notifications showing detection alerts
@app.route('/notifications')
def notifications():
    import os
    from datetime import datetime

    folder_path = 'runs/detect'
    
    if not os.path.exists(folder_path):
        return render_template('notifications.html', alerts=[])

    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    alerts = []
    latitude = session.get('user_lat')
    longitude = session.get('user_lon')
    map_link = f"https://maps.google.com/?q={latitude},{longitude}" if latitude and longitude else "Location unavailable"
    map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "https://www.google.com/maps"

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        images = [
            os.path.join(subfolder_path, img)
            for img in os.listdir(subfolder_path)
            if img.lower().endswith(('.jpg', '.png'))
        ]
        for img in images:
            relative_path = os.path.relpath(img, folder_path).replace('\\','/')
            alert = {
                "image_url": '/detections/' + relative_path,
                "time": datetime.fromtimestamp(os.path.getctime(img)).strftime("%Y-%m-%d %I:%M %p"),
                "map_link": map_link
            }
            alerts.append(alert)
    
    alerts.sort(key=lambda x: x["time"], reverse=True)
    
    return render_template('notifications.html', alerts=alerts)

# New Route to Serve Detection Images
@app.route('/detections/<path:subpath>')
def detections(subpath):
    # Build the absolute path to the requested file
    full_path = os.path.join(app.root_path, 'runs', 'detect', subpath)

    # If the file does not exist, return 404
    if not os.path.isfile(full_path):
        abort(404)

    # send_file will automatically set the correct MIME type
    return send_file(full_path)
# Renamed Display Route to avoid catching other paths
@app.route('/display/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not subfolders:
        return "No detections available.", 404
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)
    print("printing directory: ", directory) 
    files = os.listdir(directory)
    if not files:
        return "No files in the latest detection folder.", 404
    latest_file = files[0]
    print(latest_file)
    filename = os.path.join(folder_path, latest_subfolder, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory, latest_file, environ)
    else:
        return "Invalid file format"
        
def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

# Function to display the detected objects video on HTML page
@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
# Function to start the CCTV camera in realtime and detect guns and knives
@app.route("/cctv_feed")
def cctv_feed():
    # a) Load YOLOv11 once
    model = YOLO('yolo12.pt')

    # b) Open default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Could not open camera", 500

    # c) Capture base URL (still in request context)
    base = request.url_root.rstrip('/')

    def generate():
        last_alert = 0.0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # d) Run inference & annotate
            results   = model(frame, verbose=False)
            annotated = results[0].plot()
            now_ts    = time.time()

            # e) On gun/knife, max one alert every 30s
            for r in results:
                for box in r.boxes:
                    cls_idx    = int(box.cls.item())
                    class_name = model.names[cls_idx]
                    confidence = box.conf.item()

                    if class_name in ('gun', 'knife') and (now_ts - last_alert) > 30:
                        # i) Create a timestamped folder
                        folder  = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_dir = os.path.join(app.root_path, 'runs', 'detect', folder)
                        os.makedirs(out_dir, exist_ok=True)

                        # ii) Save annotated image
                        filename  = f"{class_name}_{int(now_ts)}.jpg"
                        file_path = os.path.join(out_dir, filename)
                        cv2.imwrite(file_path, annotated)

                        # iii) Build its public URL
                        img_url = f"{base}/detections/{folder}/{filename}"

                        # iv) (Optional) send SMS/WhatsApp alert
                        message = (
                            f"🚨 LIVE DETECTION 🚨\n"
                            f"Class: {class_name.upper()}\n"
                            f"Confidence: {confidence*100:.2f}%\n"
                            f"Location: {get_current_location_message()}\n"
                            f"Image: {img_url}"
                        )
                        send_whatsapp_via_twilio("+263780517601", message, img_url)

                        last_alert = now_ts
                        break  # only one alert per frame

            # f) Stream MJPEG frame
            ret, buf    = cv2.imencode('.jpg', annotated)
            frame_bytes = buf.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )

        cap.release()
        cv2.destroyAllWindows()

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def get_current_location_message():
    try:
        lat = session.get('user_lat')
        lon = session.get('user_lon')
        if lat and lon:
            return f"https://maps.google.com/?q={lat},{lon}"
        return "Location unavailable"
    except RuntimeError:
        return "Location context unavailable"

# Route for concealed detection image upload
@app.route('/concealed_detection', methods=['GET', 'POST'])
def concealed_detection():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, secure_filename(f.filename))
            f.save(filepath)
            
            try:
                # Load your YOLOv12 model
                model = YOLO('concealed.pt')  # Make sure model is in the root directory
                
                # Run inference with YOLOv12 format
                results = model.predict(source=filepath, imgsz=640, conf=0.5)
                
                # Process results
                img = cv2.imread(filepath)
                
                # Extract detections (YOLOv12 format)
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Get confidence and class
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])
                        class_label = model.names[class_id]
                        
                        # Draw bounding box and label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_label}: {confidence:.2f}"
                        cv2.putText(img, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save and return result
                output_path = os.path.join(upload_folder, 'result.jpg')
                cv2.imwrite(output_path, img)
                return send_file(output_path, mimetype='image/jpeg')
                
            except Exception as e:
                print(f"Error during inference: {e}")
                return "An error occurred during detection", 500
                
    return render_template('index.html')

# Route to Serve Concealed Images 
@app.route('/concealed/<path:filename>')
def concealed(filename):
    uploads_folder = os.path.join(os.getcwd(), 'uploads')
    return send_from_directory(uploads_folder, filename, request.environ)

# Favicon route to prevent errors
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', request.environ, mimetype='image/vnd.microsoft.icon')
    
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_effect = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, thermal_effect)
        try:
            result = CLIENT.infer(temp_path, model_id="shield-ai/2")
            print("Inference result:", result)
            for pred in result['predictions']:
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])
                confidence = pred['confidence']
                class_label = pred['class']
                cv2.rectangle(thermal_effect, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{class_label}: {confidence:.2f}"
                cv2.putText(thermal_effect, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error during inference: {e}")
        ret, buffer = cv2.imencode('.jpg', thermal_effect)
        if not ret:
            break
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Start thermal camera feed
@app.route('/start_thermal_camera')
def start_thermal_camera():
    return Response(generate_thermal_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Code for atlas assistant
@app.route('/atlas')
def atlas():
    return render_template('atlas.html')

# Privacy policy page
@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

# Delete alert route
@app.route('/delete_alert', methods=['POST'])
def delete_alert():
    data = request.get_json()
    image_url = data.get("image_url")
    
    if not image_url:
        return jsonify({"success": False, "error": "No image URL provided"}), 400

    prefix = "/detections/"
    if image_url.startswith(prefix):
        relative_path = image_url[len(prefix):]
    else:
        return jsonify({"success": False, "error": "Invalid image URL"}), 400

    file_path = os.path.join(os.getcwd(), 'runs', 'detect', relative_path)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "error": "File not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
