import cv2
import numpy as np
import time
from djitellopy import Tello

# initialize Tello

tello = Tello()
tello.connect()
tello.streamon()
print(tello.get_battery())
#tello.takeoff()

# Global variables

isObstacle = False
obstacles = []
label = ''

# if there are any obstacles found change the variable
def changeObstacle():
    global isObstacle
    if isObstacle:
        isObstacle = False
    else:
        isObstacle = True
    return isObstacle


def detectObject():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")  # Original yolov3
    classes = []
    global label
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(classes)

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # if we want to use the webcam
    # cap = cv2.VideoCapture(0)

    cap = tello.get_frame_read().cap
    font = cv2.FONT_HERSHEY_PLAIN
    frame_id = 0
    start_time = time.time()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        # in case of wabcam use tello.cap
        #_, frame = cap.read()
        frame = tello.get_frame_read().frame
        frame_id += 1

        channels = frame.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (160, 160), (0, 0, 0), True, crop=False)  # reduce 416 to 320
        net.setInput(blob)
        outs = net.forward(outputlayers)

        # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # rectangle co-ordinaters
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(
                        float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                print(label)

                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)

                if len(label) > 0:
                    changeObstacle()

        elapsed_time = time.time() - start_time
        fps = frame_id / elapsed_time
        print("Number of frame: " + str(frame_id))
        print(elapsed_time)
        cv2.putText(frame, "FPS:" + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)

        if isObstacle:
            cv2.putText(frame, "We have found an obstacle:  " + label, (10, 80), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, "To avoid the obstacle press A for LEFT, D for RIGHT movement", (10, 120), font, 1,
                        (255, 255, 255), 2)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)  # wait 1ms the loop will start again, and we will process the next frame

        if key == 27:  # esc key stops the process
            break
        elif key == ord('a'):
            avoid(obstacles, 'left')  # avoid to left
        elif key == ord('d'):
            avoid(obstacles, 'right')  # avoid to right
    cap.release()
    cv2.destroyAllWindows()


def avoid(obstacles, direction):
    if isObstacle:
        if direction == 'right':
            tello.move('right', 25)
            tello.move('forward', 75)
            tello.move('left', 25)
            print('avoid to left')
        elif direction == 'left':
            tello.move('left', 25)
            tello.move('forward', 75)
            tello.move('right', 25)
