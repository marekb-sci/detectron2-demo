# -*- coding: utf-8 -*-

import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import cv2

from detectron2 import model_zoo

from detection_tools import create_predictor, predict_and_visualize


app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

detection_config = dict(
    config_file = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"),
    model_weights = 'models/mask_rcnn_R_50_FPN_3x/model_final_f10217.pkl',
    object_classes = [2,5,7], #{0: 'person', 1:'bicycle', 2:'car', 3:'motorcycle', 4:'airplane', 5:'bus', 6:'train', 7:'truck'}
    detection_threshold = 0.5
    )

predictor = create_predictor(detection_config['config_file'], detection_config['model_weights'])


@app.route('/')
def index():
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def make_predictions():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        filepath_upload = os.path.join(app.config['UPLOAD_PATH'], filename)
        uploaded_file.save(filepath_upload)
        result = predict_and_visualize(
            filepath_upload, predictor,
            detection_config['object_classes'],
            detection_config['detection_threshold']
            )

        filepath_out = os.path.join('static', 'results.jpg')
        cv2.imwrite(filepath_out, result['image'])

    return render_template("results.html", user_image = filepath_out, n_detections=result['n_preds'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)

