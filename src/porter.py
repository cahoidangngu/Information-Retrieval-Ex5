"""Porter (1980) stemmer, implemented from scratch (nltk is not an allowed dependency).

Classic algorithm, steps 1a-5b; verified in tests/test_porter.py on the examples from Porter's paper.
"""
from functools import lru_cache


def _cons(w, i):
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(stem):
    """Measure: number of VC sequences in [C](VC)^m[V]."""
    n = len(stem); i = 0; m = 0
    while i < n and _cons(stem, i):
        i += 1
    while i < n:
        while i < n and not _cons(stem, i):
            i += 1
        if i >= n:
            break
        m += 1
        while i < n and _cons(stem, i):
            i += 1
    return m


def _has_vowel(stem):
    return any(not _cons(stem, i) for i in range(len(stem)))


def _dbl_cons(w):
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w):
    if len(w) < 3:
        return False
    return _cons(w, len(w) - 3) and not _cons(w, len(w) - 2) and _cons(w, len(w) - 1) and w[-1] not in "wxy"


def _replace(w, suffix, repl, min_m):
    """If w ends with suffix and measure of the stem > min_m, replace. Returns (word, matched_suffix)."""
    if w.endswith(suffix):
        stem = w[: len(w) - len(suffix)]
        if _m(stem) > min_m:
            return stem + repl, True
        return w, True
    return w, False


_STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
          ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
          ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
          ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
          ("logi", "log")]
_STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")]
_STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism",
          "ate", "iti", "ous", "ive", "ize"]


@lru_cache(maxsize=None)
def stem(word):
    w = word.lower()
    if len(w) <= 2:
        return w
    # step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # step 1b
    flag = False
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w = w[:-2]; flag = True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w = w[:-3]; flag = True
    if flag:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _dbl_cons(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _m(w) == 1 and _cvc(w):
            w += "e"
    # step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    # step 2
    for suf, rep in _STEP2:
        w2, hit = _replace(w, suf, rep, 0)
        if hit:
            w = w2; break
    # step 3
    for suf, rep in _STEP3:
        w2, hit = _replace(w, suf, rep, 0)
        if hit:
            w = w2; break
    # step 4
    for suf in _STEP4:
        if w.endswith(suf):
            stem_ = w[: len(w) - len(suf)]
            if _m(stem_) > 1 and (suf != "ion" or (stem_ and stem_[-1] in "st")):
                w = stem_
            break
    # step 5a
    if w.endswith("e"):
        s_ = w[:-1]
        if _m(s_) > 1 or (_m(s_) == 1 and not _cvc(s_)):
            w = s_
    # step 5b
    if _m(w) > 1 and _dbl_cons(w) and w.endswith("l"):
        w = w[:-1]
    return w
