import cv2


class Cammentator:

    def mark_faces(self, frame):
        haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = haar_cascade_face.detectMultiScale(frame)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = self.mark_eyes(frame, x, y, w, h)

        return frame

    def mark_eyes(self, frame, x, y, w, h):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faceROI = frame_gray[y:y + h, x:x + w]

        # -- In each face, detect eyes
        eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

        return frame

    def start(self):
        cv2.namedWindow("preview")
        webcam = cv2.VideoCapture(1)

        if webcam.isOpened():  # try to get the first frame
            rval, frame = webcam.read()
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = webcam.read()
            frame = self.mark_faces(frame)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cv2.destroyWindow("preview")