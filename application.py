#Flask related imports
from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import os.path
import os
import utils.centroid_tracker as centroid


application = app = Flask(__name__)
app.secret_key = b'_51123*1823*&!*y2L"F4Q8z\n\xec]/'

ENV = 'dev'

if ENV == 'dev':
    app.debug = True

else:
    app.debug = False

# File with the prediction boxes
output_file_name = 'static/boxes.jpg'

# Initialize the Centroid Tracker
ct = centroid.CentroidTracker()

# Detector Params
confidence_treshold = 0.5  # Confidence Threshold
nms_treshold = 0.4   # Non max suppression threshold
img_width = 416       # Image Width to feed the network
img_height = 416      # Image Height to feed the network

# Load Classes
coco_classes_name = "dnn_config_files/coco.names"
classes = None
with open(coco_classes_name, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load model and configuration file
model_configuration = "dnn_config_files/yolov3.cfg"
model_weights = "dnn_config_files/yolov3.weights"   
net = cv.dnn.readNetFromDarknet(model_configuration, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Configurations for image uploading
DEV_PATH = '/app/static/images/uploads'
app.config["IMAGE_UPLOADS"] = 'static/images/uploads'
app.config["IMAGE_EXTENSIONS"] = ["JPG", "JPEG"]
app.config['MAX_FILESIZE'] = 2.5 * 1024 * 1024



def get_output_names(net):
    layer_names = net.getLayerNames()

    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def draw_prediction(frame, class_id, conf, left, top, right, bottom):
    if class_id == 0:
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        label = '%.2f' % conf
            

        if classes:
            assert(class_id < len(classes))
            label = '%s:%s' % (classes[class_id], label)
    
    
def post_process(frame, outs):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_treshold and class_id == 0:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
 
    indexes = cv.dnn.NMSBoxes(boxes, confidences, confidence_treshold, nms_treshold)
    counter = 0
    for i in indexes:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_prediction(frame, class_ids[i], confidences[i], left, top, left + width, top + height)
        counter+=1
    return boxes

def allowed_image(filename):
    if not "." in filename:
        return False
    
    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

def allowed_size(filesize):
    if int(filesize) <= app.config['MAX_FILESIZE']:
        return True
    else:
        return False

def count_boxes(boxes):    
    count = 0
    for i in boxes:
        count +=1
    return count


def confidence(outs):
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_treshold:
                confidences.append(float(confidence))
    if len(confidences) > 0:
        res = np.median(confidences)
        
        return round(res, 2)
    else:
        return 1


def do_predictions():
    try:
        cap = cv.VideoCapture("static/images/uploads/input")

        contains_frame, frame = cap.read()

        if not contains_frame:
            return [False, 0, 0, 'No Frame Found']
        
        img_blob = cv.dnn.blobFromImage(frame, 1/255, (img_width, img_height), [0,0,0], 1, crop=False)
        
        net.setInput(img_blob)
        
        outs = net.forward(get_output_names(net))

        caixas = post_process(frame, outs)
        
        num_of_objects = count_boxes(caixas)
        
        t, _ = net.getPerfProfile()
        
        conf = confidence(outs)

        cv.imwrite(output_file_name, frame.astype(np.uint8))

        cap.release()

        return_obj = [True, num_of_objects, conf, None]

        return return_obj
    except:
        try:
            if cap:
                cap.release()
        except: 
            return [False, 0, 0, 'Failed to detect objects, exiting the process....']
        return [False, 0, 0, 'Failed to detect objects, exiting the process....']


@app.route('/v1')
def index():
    data = {
            "status": 200,
            "message": "Welcome to the PTSafe flask API"
    }
    return jsonify(data)
  

@app.route('/v1/predict', methods=["POST"])
def upload_image():
    if "file" not in request.files:
        data = {
            "status": 400,
            "message": "File does not exist"
        }
        return jsonify(data)
    file = request.files["file"]
    
    # If the user does not select a file, then return 400 with message
    if file.filename == "":
        data = {
            "status": 400,
            "message": "No image selected for uploading"
        }
        return jsonify(data)
    
    if not allowed_image(file.filename):
        data = {
			"status": 400,
			"message": "incorrect image format for uploading"
		}
        return jsonify(data)
    file_name = 'input'
    if os.path.exists(file_name):
        os.remove(file_name)
    file.save(os.path.join(app.config["IMAGE_UPLOADS"], file_name))
    suc, num, conf, error_message = do_predictions()
    data = {
		"status": 200,
		"number_people": num,
		"percentage_people": num / 84,
        "confidence": round(conf*100,1),
        "suc": suc, 
        "errorMsg": error_message
	}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)