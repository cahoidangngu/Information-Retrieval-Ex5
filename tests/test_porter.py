"""Porter stemmer checked on examples from Porter (1980)."""
import pytest
from porter import stem

CASES = {
    # step 1a/1b/1c
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat", "feed": "feed",
    "agreed": "agre", "plastered": "plaster", "bled": "bled", "motoring": "motor", "sing": "sing",
    "conflated": "conflat", "troubled": "troubl", "sized": "size", "hopping": "hop", "tanned": "tan",
    "falling": "fall", "hissing": "hiss", "fizzed": "fizz", "failing": "fail", "filing": "file",
    "happy": "happi", "sky": "sky",
    # step 2
    "relational": "relat", "conditional": "condit", "rational": "ration", "valenci": "valenc",
    "digitizer": "digit", "operator": "oper", "feudalism": "feudal", "decisiveness": "decis",
    "hopefulness": "hope", "callousness": "callous", "formaliti": "formal", "sensitiviti": "sensit",
    # step 3-5
    "triplicate": "triplic", "formative": "form", "formalize": "formal", "electriciti": "electr",
    "electrical": "electr", "hopeful": "hope", "goodness": "good", "revival": "reviv", "allowance": "allow",
    "inference": "infer", "airliner": "airlin", "gyroscopic": "gyroscop", "adjustable": "adjust",
    "defensible": "defens", "irritant": "irrit", "replacement": "replac", "adjustment": "adjust",
    "dependent": "depend", "adoption": "adopt", "communism": "commun", "activate": "activ",
    "effective": "effect", "bowdlerize": "bowdler", "probate": "probat", "rate": "rate", "cease": "ceas",
    "controll": "control", "roll": "roll",
}


@pytest.mark.parametrize("word,expected", list(CASES.items()))
def test_porter_examples(word, expected):
    assert stem(word) == expected


def test_short_words_untouched_and_domain_words():
    assert stem("is") == "is" and stem("a") == "a"
    assert stem("diets") == stem("diet") and stem("cancers") == stem("cancer")
    assert stem("vitamins") == stem("vitamin")
