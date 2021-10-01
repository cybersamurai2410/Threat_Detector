import cv2
import numpy as np
from imutils.video import FPS
from datetime import datetime


class Vision(object):

    def __init__(self):

        self._net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self._classes = []
        self.label = None

    def load_yolo(self):
        with open("obj.names", "r") as f:
            self._classes = [line.strip() for line in f.readlines()]

        layers_names = self._net.getLayerNames()
        output_layers = [layers_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]
        colours = np.random.uniform(0, 255, size=(len(self._classes), 3))

        return self._net, self._classes, colours, output_layers

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        return img, height, width, channels

    def detect_objects(self, img, net, outputLayers):
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(outputLayers)

        return blob, outputs

    def get_box_dimensions(self, outputs, height, width):
        boxes = []
        confs = []
        class_ids = []

        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                # Confidence threshold
                if conf > 0.3:
                    # print(conf, class_id)

                    # Object detected
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)

                    # Box coordinates
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confs.append((float(conf)))
                    class_ids.append(class_id)

        return boxes, confs, class_ids

    def draw_labels(self, boxes, confs, colours, class_ids, classes, img, path):
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                self.label = str(classes[class_ids[i]])
                colour = colours[i]

                text = "%s [%.2f]" % (self.label, max(confs))
                cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(img, text, (x, y - 5), font, 1, colour, 1)

                print(confs)

        # self.set_alert(img, path) if confs else None
        # self.set_status(img)

        img = cv2.resize(img, (500, 500))
        cv2.imshow("Threat Detector", img)

    def set_status(self, img):
        if self.label:
            print("Possession of ", self.label, " in area; take necessary precautions.")
            cv2.putText(img, "Status: {}".format('Danger'), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(img, "Status: {}".format('Safe'), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    def set_alert(self, img, path):
        print("ALERT! ---> Threat detected and captured ...")
        if path.endswith('jpg'):
            filename = path.replace("Images", "")
            cv2.imwrite('Predictions/'+filename, img)
            # send image to email and sound alarm
        else:
            cv2.imwrite('Predictions/'+path, img)


class Detector(object):

    def __init__(self):
        self._vision = Vision()
        self.model, self.classes, self.colours, self.output_layers = self._vision.load_yolo()

    def image_detect(self, img_path):
        image, height, width, channels = self._vision.load_image(img_path)
        blob, outputs = self._vision.detect_objects(image, self.model, self.output_layers)
        boxes, confs, class_ids = self._vision.get_box_dimensions(outputs, height, width)
        self._vision.draw_labels(boxes, confs, self.colours, class_ids, self.classes, image, img_path)

        while True:
            key = cv2.waitKey(1)
            if key == 27:
                break

    def video_detect(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = FPS().start()

        speed = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        filename = video_path.replace("Videos", "")
        video_writer = cv2.VideoWriter('Predictions/Video_Analysis/' + filename,
                                       cv2.VideoWriter_fourcc(*'MPV4'), speed, size)
        # black bars in frame to show text

        print("Processing video analysis ...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # show loading signal while video still processing
            print("...")

            # process video in grayscale then play back in normal format
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display date and time of surveillance footage.
            # cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (0, 475),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            height, width, channels = frame.shape
            blob, outputs = self._vision.detect_objects(frame, self.model, self.output_layers)
            boxes, confs, class_ids = self._vision.get_box_dimensions(outputs, height, width)
            self._vision.draw_labels(boxes, confs, self.colours, class_ids, self.classes, frame, filename)
            video_writer.write(frame)

            key = cv2.waitKey(1)
            fps.update()
            if key == 27:
                break

        fps.stop()
        print("Video analysis complete:")
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cap.release()

    def webcam_detect(self):
        cap = cv2.VideoCapture(0)

        while True:
            _, frame = cap.read()

            # Display date and time of live footage.
            cv2.putText(frame, str(datetime.now()), (0, 475),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Capture image when threat detected from live feed.
            dt = datetime.now()
            dt = dt.strftime("(%d %m %Y %H %M %S)")
            filename = "[Threat] " + dt + "_.jpg"
            print(filename)

            height, width, channels = frame.shape
            blob, outputs = self._vision.detect_objects(frame, self.model, self.output_layers)
            boxes, confs, class_ids = self._vision.get_box_dimensions(outputs, height, width)
            self._vision.draw_labels(boxes, confs, self.colours, class_ids, self.classes, frame, filename)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
