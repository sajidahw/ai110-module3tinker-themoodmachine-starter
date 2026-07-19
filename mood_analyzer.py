# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Emoticons (":)", ":-(", ";d") and emoji unicode ranges (pictographs, misc
# symbols/dingbats, regional indicators). Matched against already-lowercased
# text, so uppercase emoticon letters like ":D" won't appear here.
_EMOJI_OR_EMOTICON = re.compile(
    r"(:-?\)|:-?\(|:'\(|:-?d|:-?p|;-?\)|"
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF])"
)

# Collapses runs of 3+ identical characters down to 2 (e.g. "soooo" ->
# "soo"). Stopping at 2 (not 1) avoids corrupting real double letters
# like "good" or "happy", which never repeat a letter 3+ times.
_REPEATED_CHARS = re.compile(r"(.)\1{2,}")

# Words that flip the sentiment of the token immediately following them
# (e.g. "not happy", "never fun"). Only the directly adjacent word is
# flipped -- "not very happy" won't negate "happy" since "very" sits
# between them.
_NEGATION_WORDS = {
    "not", "no", "never", "cannot",
    "don't", "doesn't", "didn't", "won't", "wouldn't",
    "can't", "couldn't", "isn't", "wasn't", "aren't", "weren't",
}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()

        # Separate emojis/emoticons from adjacent words or other emojis
        # (e.g. "good:)" or "😂🔥") so each becomes its own token.
        cleaned = _EMOJI_OR_EMOTICON.sub(lambda m: f" {m.group(0)} ", cleaned)

        tokens = cleaned.split()

        # Strip leading/trailing punctuation from word tokens (e.g. "day."
        # -> "day"), but leave emoticon tokens like ":)" untouched since
        # they're built entirely out of punctuation characters.
        tokens = [
            token if _EMOJI_OR_EMOTICON.fullmatch(token) else token.strip(string.punctuation)
            for token in tokens
        ]
        tokens = [token for token in tokens if token]

        # Normalize elongated slang (e.g. "soooo" -> "soo"), skipping
        # emoticon tokens so they aren't altered.
        tokens = [
            token if _EMOJI_OR_EMOTICON.fullmatch(token) else _REPEATED_CHARS.sub(r"\1\1", token)
            for token in tokens
        ]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _score_and_hits(self, text: str) -> Tuple[int, int, int]:
        """
        Shared scoring pass used by both score_text() and predict_label().

        Returns (score, positive_hits, negative_hits), where positive_hits
        and negative_hits count contributions AFTER negation is applied
        (e.g. "not bad" counts as a positive hit, not a negative one).
        Tracking both hit counts (not just the net score) is what lets
        predict_label() tell a truly neutral post ("This is fine") apart
        from one with cancelling positive and negative signals ("mixed").
        """
        tokens = self.preprocess(text)

        score = 0
        positive_hits = 0
        negative_hits = 0
        skip_next = False
        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue

            if token in _NEGATION_WORDS and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token in self.positive_words:
                    score -= 1
                    negative_hits += 1
                    skip_next = True
                    continue
                elif next_token in self.negative_words:
                    score += 1
                    positive_hits += 1
                    skip_next = True
                    continue

            if token in self.positive_words:
                score += 1
                positive_hits += 1
            elif token in self.negative_words:
                score -= 1
                negative_hits += 1

        return score, positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        score, _, _ = self._score_and_hits(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The mapping is:
          - both positive and negative words present -> "mixed"
          - score > 0                                -> "positive"
          - score < 0                                -> "negative"
          - score == 0, no sentiment words found      -> "neutral"

        "mixed" is checked first so that cancelling signals (e.g. one
        positive word and one negative word, net score 0) are told apart
        from truly neutral text with no sentiment words at all.
        """
        score, positive_hits, negative_hits = self._score_and_hits(text)

        if positive_hits > 0 and negative_hits > 0:
            return "mixed"
        elif score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
