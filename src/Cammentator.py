import cv2

from src.ObjectDetector import ObjectDetector


class Cammentator:

    def start(self):
        cv2.namedWindow("preview")
        webcam = cv2.VideoCapture(1)

        if webcam.isOpened():  # try to get the first frame
            rval, frame = webcam.read()
        else:
            rval = False

        detector = ObjectDetector()

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = webcam.read()
            detector.mark_faces(frame)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cv2.destroyWindow("preview")