

import numpy as np
import tensorflow as tf
import os
from collections import defaultdict
from sqlalchemy.orm import sessionmaker
from tabledef import *

engine = create_engine('sqlite:///login_db.db', echo=True)

import pickle
import cv2
import os.path
from PIL import Image
import flask
from flask import request, url_for, Response
from flask import flash, redirect, render_template, request, session, abort
import requests
import io

# initialize the Flask application and other variables
app = flask.Flask(__name__)
app.secret_key = os.urandom(1)

model = None
user_db = None
IMAGE_SAVE_PATH = './images'


from lib.src.align import detect_face
from utils import (
    load_model,
    get_face,
    get_faces_live,
    forward_pass,
    save_embedding,
    load_embeddings,
    identify_face,
    allowed_file,
    remove_file_extension,
    save_image
)
model_path = 'model/20180402-114759/20180402-114759.pb'
facenet_model = load_model(model_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
image_size = 160
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# Initiate persistent FaceNet model in memory
facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

# Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)


# for checking whether a face is there in image or not
# returns true if a face is found and also saves a cropped bounded face picture 
# otherwise returns false
def face_present(image_path):
    img = cv2.imread(image_path, -1)
    save_loc = 'saved_image/new.jpg'
    face_present = False

    faces, rects = get_faces_live(
	    img=img,
	    pnet=pnet,
	    rnet=rnet,
	    onet=onet,
	    image_size=image_size
    )
    
    for face_img in faces:
        cv2.imwrite(save_loc, face_img)
        face_present = True
        cv2.imwrite('static/saved_images/bounded.jpg', face_img)
    return face_present

# load the saved user database
def ini_user_database():
    global user_db
    # check for existing database
    if os.path.exists('database/user_dict.pickle'):
        with open('database/user_dict.pickle', 'rb') as handle:
            user_db = pickle.load(handle)
    else:
        # make a new one
        # we use a dict for keeping track of mapping of each person with his/her face encoding
        user_db = defaultdict(dict)
    return user_db

# for checking if the given input face is of a registered user or not
def face_recognition(encoding, database, model, threshold=0.6):
    min_dist = 99999
    # keeps track of user authentication status
    authenticate = False
    # loop over all the recorded encodings in database
    ii = 0
    for email in database.keys():
        if ii == 0:
            ii = 1
            continue
        # find the similarity between the input encodings and claimed person's encodings using L2 norm
        dist = np.linalg.norm(np.subtract(database[email]['encoding'], encoding))
        # check if minimum distance or not
        if dist < min_dist:
            min_dist = dist
            identity = email

    if min_dist > threshold:
        print("User not in the database.")
        identity = 'Unknown Person'
        authenticate = False
    else:
        print("Hi! " + str(identity) + ", L2 distance: " + str(min_dist))
        authenticate = True

    return min_dist, identity, authenticate

# for adding user face
def add_face():
    data = {"face_present": False}
    encoding = None
    # CHECK FOR FACE IN THE IMAGE
    valid_face = False
    valid_face = face_present('saved_image/new.jpg')
    # add user only if there is a face inside the picture
    if valid_face:
        # create image encoding 
        # encoding = img_to_encoding('saved_image/new.jpg', model)
        face_img = cv2.imread('saved_image/new.jpg')
        encoding = forward_pass(
	        img=face_img,
	        session=facenet_persistent_session,
	        images_placeholder=images_placeholder,
	        embeddings=embeddings,
	        phase_train_placeholder=phase_train_placeholder,
	        image_size=image_size
        )
        # save the output for sending as json
        data['face_present'] = True
    else:
        # save the output for sending as json
        data['face_present'] = False
        print('No subject detected !')
    
    return data, encoding

def get_question_papers():

    questions = ['Are you familiar with facial recognition/do you know facial recognition?'
        , 'Have you used facial recognition before?'
        , 'On what platforms have you used facial recognition?'
        , 'How would you rate their accuracy?'
        , 'Did you pay to use the facial recognition technology?']
    answers = [['Yes', 'No']
        , ['Yes', 'No']
        , ['Social media', 'Home security', 'Work security', 'Other security required by the State']
        , ['Above average', 'Average', 'Below average', 'Very poor']
        , ['Yes', 'No']]
    question_details = {'question': questions, 'answer': answers}
    return question_details

#dashboard page
@app.route('/dashboard')
def dashboard():
    return flask.render_template('dashboard.html')

# index page
@app.route('/')
def index():
    # return render_template('questions.html', question_papers=get_question_papers())

    if not session.get('logged_in'):
        return flask.render_template("index.html")
    else:
        return dashboard()

# login page
@app.route('/login')
def login():
    return flask.render_template("login.html", question_papers=get_question_papers())


# for verifying user
@app.route('/authenticate_user', methods=["POST"])
def authenticate_user():
    POST_USERNAME = str(request.form['exampleInputEmail1'])
    POST_PASSWORD = str(request.form['exampleInputPassword1'])
    #making a session
    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User).filter(User.username.in_([POST_USERNAME]), User.password.in_([POST_PASSWORD]) )
    result = query.first()
    # if the user is logged in
    if result:
        session['logged_in'] = True
        return dashboard()
    else:
        flash('wrong password!')
    return login()

#logout page
@app.route("/logout", methods=['POST'])
def logout():
    # logging out the user
    session['logged_in'] = False
    return index()

# Sign up page display
@app.route('/sign_up')
def sign_up():
    return flask.render_template("sign_up.html")

# to add user through the sign up from
@app.route('/signup_user', methods=["POST"])
def signup_user():
    #declaring the engine
    engine = create_engine('sqlite:///login_db.db', echo=True)
    
    # whether user registration was successful or not
    user_status = {'registration': False, 'face_present': False, 'duplicate':False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print('Inside post')
        # getting the email and password from the user
        POST_USERNAME = str(request.form['email'])
        POST_PASSWORD = str(request.form['pass'])
        NAME = str(request.form['name'])

        if POST_USERNAME not in user_db.keys():
            # add new user's face
            if flask.request.files.get("image"):
                print('Inside Image')
                # read the image in PIL format
                image = flask.request.files["image"].read()
                image = np.array(Image.open(io.BytesIO(image)))
                print('Image saved success')
                # save the image on server side
                cv2.imwrite('saved_image/new.jpg',
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # check if any face is present or not in the picture
                data, encoding = add_face()
                # set face detected as True
                user_status['face_present'] = data['face_present']
            # if no image was sent
            else:
                user_status['face_present'] = False

            # only create a new session if complete user details is present
            if data['face_present']:
                # create a new session
                Session = sessionmaker(bind=engine)
                s = Session()
                # add data to user_db dict
                user_db[POST_USERNAME]['encoding'] = encoding
                user_db[POST_USERNAME]['name'] = NAME

                # save the user_db dict
                with open('database/user_dict.pickle', 'wb') as handle:
                    pickle.dump(user_db, handle,protocol=pickle.HIGHEST_PROTOCOL)
                print('User ' + POST_USERNAME + ' added successfully')

                # adding the user to data base
                user = User(POST_USERNAME, POST_PASSWORD)
                s.add(user)
                s.commit()

                # set registration status as True
                user_status['registration'] = True
                #logging in the user
                session['logged_in'] = True
                #return dashboard()
        else:
            user_status['duplicate'] = True
    
    #return sign_up()
    return flask.jsonify(user_status)
   
# predict function 
@app.route("/predict", methods=["POST"])
def predict():
    # this will contain the 
    data = {"success": False}
    # for keeping track of authentication status
    data['authenticate'] = False
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = np.array(Image.open(io.BytesIO(image)))
            
            # save the image on server side
            cv2.imwrite('saved_image/new.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # CHECK FOR FACE IN THE IMAGE
            valid_face = False
            valid_face = face_present('saved_image/new.jpg')

            # do facial recognition only when there is a face inside the frame
            if valid_face:
                # find image encoding and see if the image is of a registered user or not
                # encoding = img_to_encoding('saved_image/new.jpg', model)
                face_img = cv2.imread('saved_image/new.jpg')
                encoding = forward_pass(
	                img=face_img,
	                session=facenet_persistent_session,
	                images_placeholder=images_placeholder,
	                embeddings=embeddings,
	                phase_train_placeholder=phase_train_placeholder,
	                image_size=image_size
                )
                min_dist, identity, authenticate = face_recognition(
                                                    encoding, user_db, model, threshold=0.9)
                
                # save the output for sending as json
                data["min_dist"] = str(min_dist)
                data['email'] = identity
                if identity != 'Unknown Person':
                    data['name'] = user_db[identity]['name']
                else:
                    data['name'] = 'Unknown Person'
                data['face_present'] = True
                data['authenticate'] = authenticate

            else:
                # save the output for sending as json
                data["min_dist"] = 'NaN'
                data['identity'] = 'NaN'
                data['name'] = 'NaN'
                data['face_present'] = False
                data['authenticate'] = False
                print('No subject detected !')
            
            # indicate that the request was a success
            data["success"] = True

        # create a new session
        Session = sessionmaker(bind=engine)
        s = Session()
        # check if the user is logged in
        if data['authenticate']:
            session['logged_in'] = True
        else:
            flash('Unknown Person!')

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route('/dashboard1')
def dashboard1():
    return flask.render_template('questions.html', question_papers=get_question_papers())
    # return render_template('questions.html', question_papers=get_question_papers())


# first load the model and then start the server
if __name__ == "__main__":
    

    print("** Starting Flask server.........Please wait until the server starts ")
    print('Loading the Neural Network......\n')
    ini_user_database()
    print('Database loaded...........')
    app.run(host='0.0.0.0', port=5000)
