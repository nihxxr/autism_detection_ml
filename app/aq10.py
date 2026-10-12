"""AQ-10 (adult) items and scoring.

Source: Allison C, Auyeung B, Baron-Cohen S. "Toward brief 'red flags' for
autism screening: the short Autism Spectrum Quotient and the short
Quantitative Checklist in 1,000 cases and 3,000 controls." J Am Acad Child
Adolesc Psychiatry, 2012. NICE guideline CG142 recommends a specialist
assessment referral at a score of 6 or above.
"""

from dataclasses import dataclass

ANSWERS = (
    ("definitely_agree", "Definitely agree"),
    ("slightly_agree", "Slightly agree"),
    ("slightly_disagree", "Slightly disagree"),
    ("definitely_disagree", "Definitely disagree"),
)
ANSWER_KEYS = {key for key, _ in ANSWERS}
AGREE = {"definitely_agree", "slightly_agree"}

REFERRAL_THRESHOLD = 6


@dataclass(frozen=True)
class Item:
    number: int
    text: str
    scores_on_agree: bool


ITEMS = (
    Item(1, "I often notice small sounds when others do not.", True),
    Item(2, "I usually concentrate more on the whole picture, rather than the small details.", False),
    Item(3, "I find it easy to do more than one thing at once.", False),
    Item(4, "If there is an interruption, I can switch back to what I was doing very quickly.", False),
    Item(5, "I find it easy to ‘read between the lines’ when someone is talking to me.", False),
    Item(6, "I know how to tell if someone listening to me is getting bored.", False),
    Item(7, "When I’m reading a story, I find it difficult to work out the characters’ intentions.", True),
    Item(8, "I like to collect information about categories of things, like types of car, bird, train or plant.", True),
    Item(9, "I find it easy to work out what someone is thinking or feeling just by looking at their face.", False),
    Item(10, "I find it difficult to work out people’s intentions.", True),
)


def item_point(item: Item, answer: str) -> int:
    """Return 1 if this answer scores a point for this item, else 0."""
    if answer not in ANSWER_KEYS:
        raise ValueError(f"Unknown answer {answer!r} for item {item.number}")
    agreed = answer in AGREE
    return int(agreed == item.scores_on_agree)


@dataclass(frozen=True)
class Result:
    points: tuple  # one 0/1 per item, in order
    answers: tuple

    @property
    def score(self) -> int:
        return sum(self.points)

    @property
    def above_threshold(self) -> bool:
        return self.score >= REFERRAL_THRESHOLD


def score(answers: dict) -> Result:
    """Score a full set of answers keyed q1..q10. Raises ValueError if incomplete."""
    missing = [i.number for i in ITEMS if f"q{i.number}" not in answers]
    if missing:
        raise ValueError(f"Missing answers for questions {missing}")
    picked = tuple(answers[f"q{i.number}"] for i in ITEMS)
    points = tuple(item_point(i, a) for i, a in zip(ITEMS, picked))
    return Result(points=points, answers=picked)
