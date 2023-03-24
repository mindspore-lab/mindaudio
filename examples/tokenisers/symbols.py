""" from https://github.com/keithito/tacotron """

from examples.tokenisers import cmudict

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_end = "~"
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same
# as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + list(_end)
)
