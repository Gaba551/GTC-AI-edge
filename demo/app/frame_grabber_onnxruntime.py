import argparse
import os
import numpy as np
import cv2
import time
from time import sleep
import json
import torch
from objdict import ObjDict
from datetime import datetime
from utils.general import non_max_suppression
import onnxruntime as ort

providers_cpu = [
    'CPUExecutionProvider',
]

providers_gpu = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]

# username = "joncoons"

class Cam_File_Sink():

    # def __init__(self, camID="image_file", camLocation="table_top", camPosition="side", modelFile=f"/home/{username}/demo/model-repo/gtc_onnx/1/model.onnx", labelFile=f"/home/{username}/demo/model-repo/gtc_onnx/labels.txt", targetDim="640", probThres=".2", iouThres=".4"):
    def __init__(self, username, camID, camLocation, camPosition, modelFile, labelFile, targetDim, probThres, iouThres):

        self.username = username
        self.camID = camID
        self.camLocation = camLocation
        self.camPosition = camPosition
        self.modelFile = modelFile
        self.labelFile = labelFile
        self.targetDim = int(targetDim)
        self.probThres = float(probThres)
        self.iouThres = float(iouThres)

        self.model_name = os.path.basename(modelFile)
        print(f"Model name:  {self.model_name}")
        self.frameCount = 0

        self.device_type = str.lower(ort.get_device())
        print(f"ORT device: {self.device_type}")

        with open(labelFile, 'r') as f:
            labels = [l.strip() for l in f.readlines()] 
        self.labels = labels
        if self.device_type == "cpu":
            self.session = ort.InferenceSession(modelFile, providers=providers_cpu)
        elif self.device_type == "gpu":
            self.session = ort.InferenceSession(modelFile, providers=providers_gpu)

        self.input_name = self.session.get_inputs()[0].name

        self.cap_stored_image()

    def cap_stored_image(self):
        while True:
            img_list = os.listdir(f"/home/{self.username}/demo/image_sink")
            sleep(2)
            if not img_list:
                continue
            for filename in img_list:
                if self.check_extension(filename):
                    self.cycle_begin = time.time()
                    self.frameCount += 1
                    img_path = os.path.join((f"/home/{self.username}/demo/image_sink"), filename)
                    frame = cv2.imread(img_path)
                    frame = np.asarray(frame)
                    frame_optimized = self.frame_resize(frame, self.targetDim)
                    # print(f"Frame_optimized shape = {frame_optimized.shape}")
                    frame_infer = frame_optimized.astype(np.float32)
                    frame_infer = frame_infer.transpose(2,0,1)
                    frame_infer = np.expand_dims(frame_infer, axis=0)
                    # print(f"frame_infer shape: {frame_infer.shape}")
                    frame_infer = frame_infer.astype(np.float32)/255.0 # normalize pixels
                    t1 = time.time()
                    result = self.infer_output(frame_infer)
                    t2 = time.time()
                    t_infer = (t2-t1)*1000
                    # print(f"Inference time {t_infer}ms")

                    if result:
                        # print(json.dumps(result))
                        now = datetime.now()
                        created = now.isoformat()
                        filetime = now.strftime("%Y%d%m%H%M%S%f")
                        annotatedName = f"{self.camLocation}-{self.camPosition}-{filetime}-annotated.jpg"
                        annotatedPath = os.path.join(f"/home/{self.username}/demo/images_annotated", annotatedName)
                        detection_count = len(result['predictions'])
                        print(f"Detection Count: {detection_count}")

                        inference_obj = {
                            'model_name': self.model_name,
                            'inferencing_time': t_infer,
                            'object_detected': "True",
                            'camera_id': self.camID,
                            'camera_name': f"{self.camLocation}-{self.camPosition}",
                            'annotated_image_name': annotatedName,
                            'annotated_image_path': annotatedPath,
                            'created': created,
                            'detected_objects': result['predictions']
                            }               

                        inference_message = json.dumps(inference_obj) 
                        print(f"Inference Output:  \n{inference_message}\n")                                                    

                        for i in range(detection_count):
                            tag_name = result['predictions'][i]['labelName']
                            probability = round(result['predictions'][i]['probability'],2)
                            bounding_box = result['predictions'][i]['bbox']
                            image_text = f"{tag_name}@{probability}%"
                            color = (0, 255, 0)
                            thickness = 1

                            annotated_frame = frame_optimized   

                            if bounding_box:
                                start_point = (int(bounding_box["left"]), int(bounding_box["top"]))
                                end_point = (int(bounding_box["width"]), int(bounding_box["height"]))
                                annotated_frame = cv2.rectangle(annotated_frame, start_point, end_point, color, thickness)
                                annotated_frame = cv2.putText(annotated_frame, image_text, start_point, fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = .3, color = (255,0, 0))
                            
                        self.frame_write(annotatedPath, annotated_frame)
                
                    delete_img = os.remove(img_path)
                    if delete_img:
                        print(f"Deleted image: {img_path}\n\n")

    def infer_output(self, pp_image):

        inputs = pp_image
        # if self.is_fp16:
        #     inputs = inputs.astype(np.float16)
        outputs = self.session.run(None, {self.input_name: inputs})      
        # print('Concatenated outputs shape: {}'.format(outputs[0].shape))
        # print('Separated outputs shape: {}, {}, {}'.format(outputs[1].shape, outputs[2].shape, outputs[3].shape))

        filterd_predictions = non_max_suppression(torch.tensor(outputs[0]), conf_thres = self.probThres, iou_thres = self.iouThres)
        # print(filterd_predictions)

        predictions = []

        try:
            for pred in filterd_predictions[0]: 
                x1 = round(float(pred[0]),8)
                y1 = round(float(pred[1]),8)
                x2 = round(float(pred[2]),8)
                y2 = round(float(pred[3]),8)
                probability = round(float(pred[4]),8)
                labelId = int(pred[5])
                labelName = str(self.labels[labelId])

                pred = ObjDict()
                pred.probability = float(probability*100)
                pred.labelId = int(labelId)
                pred.labelName = labelName
                pred.bbox = {
                    'left': x1,
                    'top': y1,
                    'width': x2,
                    'height': y2
                }
                predictions.append(pred)

            response = {
            'created': datetime.utcnow().isoformat(),
            'predictions': predictions
            }
            return response

        except:
            print("No predictions present")
   
    def check_extension(self, filename):
        file_extensions = set(['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'])
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in file_extensions

    def frame_resize(self, img, target):
        padColor = [0,0,0]
        h, w = img.shape[:2]
        sh, sw = (target, target)
        if h > sh or w > sw: # shrinking 
            interp = cv2.INTER_AREA
        else: # stretching 
            interp = cv2.INTER_CUBIC
        aspect = w/h  
        # compute scaling and pad sizing
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
        scaled_frame = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_frame = cv2.copyMakeBorder(scaled_frame, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)        
        return scaled_frame

    def frame_write(self, module_path, image_data):
        cv2.imwrite(module_path, image_data,[int(cv2.IMWRITE_JPEG_QUALITY), 100]) 
        return f"Successfully wrote file: {module_path}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u',
                        '--username',
                        required=True,
                        help='Add username')
    parser.add_argument('-p',
                        '--probability',
                        default=".6",
                        help='Add username')
    args = parser.parse_args()
    uname = args.username
    prob_set = args.probability
    model_args = f"/home/{uname}/demo/model-repo/gtc_onnx/1/model.onnx"
    label_args =f"/home/{uname}/demo/model-repo/gtc_onnx/labels.txt"
    
    Cam_File_Sink(
        username = uname,
        camID ="image_file",
        camLocation = "table_top", 
        camPosition = "side",
        modelFile = model_args,
        labelFile= label_args,
        targetDim = "640", 
        probThres = prob_set, 
        iouThres = ".4",
    )
        