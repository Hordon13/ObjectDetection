import cv2
import numpy as np
import wave
import pyaudio
import speech_recognition as sr

labels = open("yolo3-320/coco.names").read().strip().split("\n")


def play_effect(sound):
    chunk = 1024
    effect = None

    if sound == "start":
        effect = "media/start.wav"
    elif sound == "end":
        effect = "media/end.wav"
    elif sound == "failed":
        effect = "media/failed.wav"

    wf = wave.open(effect, 'rb')
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                     channels=wf.getnchannels(),
                     rate=wf.getframerate(),
                     output=True)

    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    pa.terminate()


def voice_command():
    rc = sr.Recognizer()
    print("I'm waiting for your commands!")
    while True:
        mic = sr.Microphone()
        with mic as source:
            rc.adjust_for_ambient_noise(source, duration=2)
            try:
                audio = rc.listen(source)
                result = rc.recognize_google(audio).lower()
                if result == "detect":
                    play_effect("start")
                    print("I am listening...")
                    audio = rc.listen(source)

                    result = rc.recognize_google(audio).lower()

                    if result in labels:
                        print("Okay, detecting: " + result)
                        play_effect("end")
                        detect_objects(result)
                    else:
                        print("Sorry, I didn't catch that.")
                        play_effect("failed")

            except:
                pass


def detect_objects(object):
    cam = cv2.VideoCapture(0)

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
