import cv2

from src.Cascades import Cascades


class ObjectDetector:

    def mark_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = Cascades.FACES.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = x, y, w, h
            self.mark_eyes(frame, face)
            self.mark_smiles(frame, face)

    def _get_face_ROI_gray(self, frame, face):
        x, y, w, h = face
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        return frame_gray[y:y + h, x:x + w]

    def _get_face_ROI_color(self, frame, face):
        x, y, w, h = face
        return frame[y:y + h, x:x + w]

    def mark_eyes(self, frame, face):
        x, y, w, h = face
        roi_gray = self._get_face_ROI_gray(frame, face)

        # -- In each face, detect eyes
        eyes = Cascades.EYES.detectMultiScale(roi_gray)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

    def mark_smiles(self, frame, face):
        x, y, w, h = face
        roi_gray = self._get_face_ROI_gray(frame, face)
        roi_color = self._get_face_ROI_color(frame, face)

        # -- In each face, detect eyes
        smiles = Cascades.SMILES.detectMultiScale(roi_gray, 1.8, 20)
        for (x2, y2, w2, h2) in smiles:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
