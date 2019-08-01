import cv2
import numpy as np



def detect_objects(object):
    cam = cv2.VideoCapture(1)

    labels = open("C:/Users/julcs/green_fox/fedex/UrbanSpotter/yolo3-320/coco.names").read().strip().split("\n")
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet("./yolo3-320/yolov3.cfg", "./yolo3-320/yolov3.weights")
    layerNames = net.getLayerNames()
    layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    (W, H) = (None, None)

    while True:
        (grabbed, frame) = cam.read()
        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (288, 288), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(layerNames)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.4:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.2)

        if len(indexes) > 0:
            for i in indexes.flatten():
                if labels[classIDs[i]] == object:

                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in colors[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.2f}".format(labels[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Breathtaking", frame)
        if cv2.waitKey(3) == 27:
            break
detect_objects("person")