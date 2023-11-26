import os
import string
import uuid
import pickle
import datetime
import time
import shutil

from flask import Flask, request, render_template, send_from_directory, jsonify
import cv2
import face_recognition

app = Flask(__name__)

ATTENDANCE_LOG_DIR = './logs'
DB_PATH = './db'
for dir_ in [ATTENDANCE_LOG_DIR, DB_PATH]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    image_data = request.form['image']
    user_name, match_status = recognize(image_data)

    if match_status:
        save_attendance_log(user_name, 'IN')

    return jsonify({'user': user_name, 'match_status': match_status})


@app.route('/logout', methods=['POST'])
def logout():
    file = request.files['file']
    user_name, match_status = recognize(file)

    if match_status:
        save_attendance_log(user_name, 'OUT')

    return jsonify({'user': user_name, 'match_status': match_status})


@app.route('/register_new_user', methods=['POST'])
def register_new_user():
    file = request.files['file']
    text = request.form['text']

    save_uploaded_file(file)
    shutil.copy(file.filename, os.path.join(DB_PATH, '{}.png'.format(text)))

    embeddings = face_recognition.face_encodings(cv2.imread(file.filename))

    file_ = open(os.path.join(DB_PATH, '{}.pickle'.format(text)), 'wb')
    pickle.dump(embeddings, file_)
    os.remove(file.filename)

    return jsonify({'registration_status': 200})


@app.route('/get_attendance_logs')
def get_attendance_logs():
    filename = 'out.zip'
    shutil.make_archive(filename[:-4], 'zip', ATTENDANCE_LOG_DIR)
    return send_from_directory('.', '{}.zip'.format(filename))


def recognize(image_data):
    # Decode the base64 image data
    image_data_decoded = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_data_decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    file_name = "recognized_image.jpg"

    # Construct the full path to the image file
    image_path = os.path.join(ATTENDANCE_LOG_DIR, file_name)
    # Save the image to the specified path with the JPEG format
    img_pil.save(image_path, format='JPEG')
    img_np = np.array(img_pil)
    # Compute facial embeddings for the image
    embeddings_unknown = face_recognition.face_encodings(img_np)
    
    if len(embeddings_unknown) == 0:
        return 'no_persons_found', False
    else:
        embeddings_unknown = embeddings_unknown[0]

    match = False
    j = 0
    db_dir = sorted([j for j in os.listdir(DB_PATH) if j.endswith('.pickle')])

    while ((not match) and (j < len(db_dir))):
        path_ = os.path.join(DB_PATH, db_dir[j])

        with open(path_, 'rb') as file:
            embeddings = pickle.load(file)[0]

        match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]

        j += 1

    if match:
        return db_dir[j - 1][:-7], True
    else:
        return 'unknown_person', False


def save_attendance_log(user_name, status):
    epoch_time = time.time()
    date = time.strftime('%Y%m%d', time.localtime(epoch_time))
    with open(os.path.join(ATTENDANCE_LOG_DIR, '{}.csv'.format(date)), 'a') as file:
        file.write('{},{},{}\n'.format(user_name, datetime.datetime.now(), status))


def save_uploaded_file(file):
    file.filename = f"{uuid.uuid4()}.png"
    file.save(file.filename)


if __name__ == '__main__':
    app.run(debug=True)
