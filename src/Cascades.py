import cv2

class Cascades:
    FACES = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    EYES = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    SMILES = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')