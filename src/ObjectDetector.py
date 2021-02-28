import cv2

from src.Cascades import Cascades


class ObjectDetector:

    def mark_faces(self, frame):
        faces = Cascades.FACES.detectMultiScale(frame)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = x, y, w, h
            frame = self.mark_eyes(frame, face)
            frame = self.mark_smiles(frame, face)

        return frame

    def _get_face_ROI(self, frame, face):
        x, y, w, h = face
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        return frame_gray[y:y + h, x:x + w]

    def mark_eyes(self, frame, face):
        x, y, w, h = face
        face_roi = self._get_face_ROI(frame, face)

        # -- In each face, detect eyes
        eyes = Cascades.EYES.detectMultiScale(face_roi)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

        return frame

    def mark_smiles(self, frame, face):
        x, y, w, h = face
        face_roi = self._get_face_ROI(frame, face)

        # -- In each face, detect eyes
        smiles = Cascades.SMILES.detectMultiScale(face_roi)
        for (x2, y2, w2, h2) in smiles:
            frame = cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

        return frame