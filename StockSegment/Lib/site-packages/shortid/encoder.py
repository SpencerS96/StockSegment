import os

class Encoder:
    def encode(self, lookup_fn, number):
        loops = 0
        res = ""

        while True:
            res = res + lookup_fn(
                ((number >> (4 * loops)) & 0x0F
            ) | (
                bytearray(os.urandom(1))[0] & 0x30)
            )
            loops += 1
            if (number < pow(16, loops)) != 0:
                break

        return res
