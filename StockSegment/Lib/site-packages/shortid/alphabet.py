import string
from random import random
from random import shuffle

class Alphabet:

    ORIGINAL = "{0}{1}_-".format(
        string.digits,
        string.ascii_letters
    )

    def __init__(self):
        self._alphabet = Alphabet.ORIGINAL
        self.reset()

    def set(self, characters):
        characters_len = len(characters)
        original_len = len(Alphabet.ORIGINAL)
        if characters_len != original_len:
            raise RuntimeError(
                "Custom alphabet for shortId must be {0} unique characters. " +
                "You submitted {1} characters: {2}".format(
                    original_len,
                    characters_len,
                    characters
                )
            )

        characters_set_len = len(set(characters))
        if characters_set_len != characters_len:
            raise RuntimeError(
                (
                    "Custom alphabet for shortId must be {0} unique characters. "
                    "{1} non unique values found."
                ).format(characters_len - characters_set_len)
            )

        self._alphabet = characters
        self.reset()

    def get(self):
        return self._alphabet

    def reset(self):
        self._shuffled = self.shuffle()

    def lookup(self, index):
        return self.get_shuffled()[index]

    def shuffle(self):
        alphabet = list(self._alphabet)
        shuffle(alphabet)
        return "".join(alphabet)

    def get_shuffled(self):
        return self._shuffled
