
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2
import random
import face_recognition

yolo_clases = {i:line for i, line in enumerate(open('clases.txt'))}


def yolo_v3(image, confidence_threshold, overlap_threshold):

    # Load the network. Because this is cached it will only happen once.
    @st.cache(allow_output_mutation=True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    # Run the YOLO neural net.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # Supress detections in case of too low confidence or too much overlap.
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)


    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = yolo_clases.get(class_IDs[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})

    return boxes


def draw_image_with_boxes(image, boxes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    rcolor = lambda x: (random.randint(0,255),random.randint(0,255),random.randint(0,255)) 
    LABEL_COLORS = {
        value: rcolor(value) for value in yolo_clases.values()
    }
    LABEL_COLORS['facee'] = rcolor('facee')
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2
        cv2.putText(image_with_boxes, label[:-1], (xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 2)

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)


def detect_faces(image):
    faces = face_recognition.face_locations(image)
    return pd.DataFrame({"xmin": face[3], "ymin": face[0], "xmax": face[1], "ymax": face[2], "labels": 'facee'} for face in faces)

path = st.text_input('Path','/home/hian/Videos/vidaocultadelosprogramadors.mpg')

yolo_clases

if path:
    vidcap = cv2.VideoCapture(path)


    @st.cache
    def get_frames(path):
        ans = []
        success, image = vidcap.read()
        count = 1
        while success:
            if count % 48 ==0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                ans.append(image)
            count += 1
            success, image = vidcap.read()

        return ans

    frames = get_frames(path)

    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, max(1,len(frames) - 1), 0)
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)

    st.image(frames[selected_frame_index])

    image = frames[selected_frame_index]

    boxes = yolo_v3(image, confidence_threshold, overlap_threshold)

    boxes

    draw_image_with_boxes(image, boxes, "Real-time Computer Vision",
        "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)" % (overlap_threshold, confidence_threshold))


    boxes = detect_faces(image)

    boxes

    draw_image_with_boxes(image, boxes, "Real-time Computer Vision",
        "Face recognition" )
