"""
Shared feature definitions for SpamShield.

The 57 input features follow the UCI Spambase schema:
  * 48 x word_freq_<word>  : 100 * (occurrences of word / total words in the email)
  *  6 x char_freq_<char>  : 100 * (occurrences of char / total characters in the email)
  *  3 x capital_run_length_{average,longest,total}
Target column: is_spam (1 = spam, 0 = legitimate mail).

`extract_features_from_text` lets the Streamlit app turn a *raw pasted email*
into the same 57-feature vector, so the saved models can score unseen mail.
"""
import re
import numpy as np

WORD_FEATURES = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference",
]
CHAR_FEATURES = [";", "(", "[", "!", "$", "#"]
CHAR_NAMES = ["semicolon", "paren", "bracket", "exclamation", "dollar", "hash"]

FEATURE_NAMES = (
    [f"word_freq_{w}" for w in WORD_FEATURES]
    + [f"char_freq_{n}" for n in CHAR_NAMES]
    + ["capital_run_length_average", "capital_run_length_longest",
       "capital_run_length_total"]
)
TARGET = "is_spam"
assert len(FEATURE_NAMES) == 57

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CAPS_RE = re.compile(r"[A-Z]+")


def extract_features_from_text(text: str) -> dict:
    """Approximate the Spambase feature extraction for a raw email string."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)

    feats = {}
    for w in WORD_FEATURES:
        feats[f"word_freq_{w}"] = 100.0 * words.count(w) / n_words
    for ch, name in zip(CHAR_FEATURES, CHAR_NAMES):
        feats[f"char_freq_{name}"] = 100.0 * text.count(ch) / n_chars

    runs = [len(r) for r in _CAPS_RE.findall(text)]
    if runs:
        feats["capital_run_length_average"] = float(np.mean(runs))
        feats["capital_run_length_longest"] = float(max(runs))
        feats["capital_run_length_total"] = float(sum(runs))
    else:
        feats["capital_run_length_average"] = 1.0
        feats["capital_run_length_longest"] = 1.0
        feats["capital_run_length_total"] = 1.0
    return feats


def log1p_transform(X):
    """Log-damps heavy-tailed frequency / run-length features.
    Lives in this shared module so saved pipelines unpickle in any entry point."""
    return np.log1p(np.asarray(X, dtype=float))
