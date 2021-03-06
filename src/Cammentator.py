import asyncio
from concurrent.futures.process import ProcessPoolExecutor

import cv2

from src.ObjectCommentator import ObjectCommentator
from src.ObjectDetector import ObjectDetector


class Cammentator:
    _commentator = ObjectCommentator()
    _webcam = cv2.VideoCapture(1)

    async def _start(self):
        task1 = asyncio.ensure_future(self._commentator.start_speaking())
        task2 = asyncio.ensure_future(self._start_detection())
        await asyncio.gather(task1, task2)

    def start(self):
        cv2.namedWindow("preview")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._start())
        loop.close()
        #asyncio.run(self._commentator.start_speaking())
        #asyncio.run(self._start_detection())

        cv2.destroyWindow("preview")

    async def _start_detection(self):
        detector = ObjectDetector(self.on_object_detected)

        if self._webcam.isOpened():  # try to get the first frame
            rval, frame = self._webcam.read()
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = self._webcam.read()
            detector.mark_faces(frame)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break

            await asyncio.sleep(0.1)

    def on_object_detected(self, object_name):
        self._commentator.new_comment(object_name)
