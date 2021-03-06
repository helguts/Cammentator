import asyncio

import pyttsx3

from src.CommentsDict import OBJECT_COMMENTS

engine = pyttsx3.init()

class ObjectCommentator:
    _engine = pyttsx3.init()
    _comments = []

    def _comment(self):
        for comment in self._comments:
            self._engine.say(comment)
            self._engine.runAndWait()

    async def start_speaking(self):
        while True:
            self._comment()
            await asyncio.sleep(0.1)

    def new_comment(self, object_name):
        if object_name in OBJECT_COMMENTS:
            self._comments.append(OBJECT_COMMENTS[object_name])


