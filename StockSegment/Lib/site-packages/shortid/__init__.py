import time
import os
from .encoder import Encoder
from .alphabet import Alphabet

class ShortId:

    def __init__(self):
        self._counter = 0
        self._previous_delta = 0

        self.reduce_delay = 86400
        self.reduce_time = (int(time.time()) * 1000) - self.reduce_delay
        self.version = 4
        self.encoder = Encoder()
        self.alphabet = Alphabet()

    def generate(self):
        delta = self._get_time_delta()

        if delta == self._previous_delta:
            self._counter += 1
        else:
            self._counter = 0
            self._previous_delta = delta

        res = self.encoder.encode(
            self.alphabet.lookup,
            self.version
        )
        res = res + self.encoder.encode(
            self.alphabet.lookup,
            os.getpid()
        )

        if self._counter > 0:
            res = res + self.encoder.encode(
                self.alphabet.lookup,
                self._counter
            )

        res = res + self.encoder.encode(
            self.alphabet.lookup,
            delta
        )

        return res

    def _get_time(self):
        return int(time.time() * 1000)

    def _get_time_delta(self):
        return int(
            round(
                (self._get_time() - self.reduce_time) * 0.01
            )
        )
