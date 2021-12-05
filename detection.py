
import numpy as  np
import cv2
import os
import random
import colorsys
print(cv2.__version__)

path_dir = 'data/'

images = [os.path.join(path_dir, img)  for img in os.listdir(path_dir) if img.endswith('.jpg')]
print(len(images))

with open('classes.txt', 'r') as f:
     class_labels = f.readlines()
     
class_labels = [cls.strip() for cls in class_labels]


img_to_detect = cv2.imread(random.choice(images))
img_height = img_to_detect.shape[0]
img_width = img_to_detect.shape[1]


img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)

hsv_tuples = [(x / len(class_labels), 1., 1.) for x in range(len(class_labels))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
class_colors = list(map(lambda x:(int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
np.random.seed(43)
np.random.shuffle(colors)
np.random.seed(None)
class_colors = np.tile(class_colors,(16,1))

yolo_model = cv2.dnn.readNetFromDarknet('weights/car.cfg','weights/car_best.weights')

yolo_layers = yolo_model.getLayerNames()
yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
yolo_model.setInput(img_blob)
obj_detection_layers = yolo_model.forward(yolo_output_layer)
class_ids_list = []
boxes_list = []
confidences_list = []
############## NMS Change 1 END ###########


# loop over each of the layer outputs
for object_detection_layer in obj_detection_layers:
	# loop over the detections
    for object_detection in object_detection_layer:
        
        # obj_detections[1 to 4] => will have the two center points, box width and box height
        # obj_detections[5] => will have scores for all objects within bounding box
        all_scores = object_detection[5:]
        predicted_class_id = np.argmax(all_scores)
        prediction_confidence = all_scores[predicted_class_id]
    
        # take only predictions with confidence more than 20%
        if prediction_confidence > 0.20:
            #get the predicted label
            predicted_class_label = class_labels[predicted_class_id]
            #obtain the bounding box co-oridnates for actual image from resized image size
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
            start_x_pt = int(box_center_x_pt - (box_width / 2))
            start_y_pt = int(box_center_y_pt - (box_height / 2))
            
            ############## NMS Change 2 ###############
            #save class id, start x, y, width & height, confidences in a list for nms processing
            #make sure to pass confidence as float and width and height as integers
            class_ids_list.append(predicted_class_id)
            confidences_list.append(float(prediction_confidence))
            boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
            ############## NMS Change 2 END ###########
            
############## NMS Change 3 ###############
# Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes      
# Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

# loop through the final set of detections remaining after NMS and draw bounding box and write text
for max_valueid in max_value_ids:
    max_class_id = max_valueid[0]
    box = boxes_list[max_class_id]
    start_x_pt = box[0]
    start_y_pt = box[1]
    box_width = box[2]
    box_height = box[3]
    
    #get the predicted class id and label
    predicted_class_id = class_ids_list[max_class_id]
    predicted_class_label = class_labels[predicted_class_id]
    prediction_confidence = confidences_list[max_class_id]

           
    end_x_pt = start_x_pt + box_width
    end_y_pt = start_y_pt + box_height
    
    #get a random mask color from the numpy array of colors
    box_color = class_colors[predicted_class_id]
    
    #convert the color numpy array as a list and apply to text and box
    box_color = [int(c) for c in box_color]
    
    # print the prediction in console
    predicted_boxes_classes_ = "{}: {:.2f}% {} {} {} {}".format(predicted_class_label, prediction_confidence * 100, start_x_pt, start_y_pt,end_x_pt, end_y_pt)
    predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
    with open('pred_cls.txt', 'a') as f:
         f.write(str(predicted_boxes_classes_)+'\n')
    print("predicted object {}".format(predicted_class_label))
    
    # draw rectangle and text in the image
    cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
    cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

image = cv2.cvtColor(img_to_detect, cv2.COLOR_BGR2RGB)
cv2.imwrite('out.jpg', img_to_detect)
cv2.imshow("Detection Output", img_to_detect)
