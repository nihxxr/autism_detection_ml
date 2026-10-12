import pytest

from app import create_app
from app import aq10


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def answers_scoring(n):
    """Build a full answer set that scores exactly n points."""
    out = {}
    for i, item in enumerate(aq10.ITEMS):
        scoring = i < n
        agree = item.scores_on_agree == scoring
        out[f"q{item.number}"] = "slightly_agree" if agree else "slightly_disagree"
    return out


def test_scoring_direction():
    item1, item2 = aq10.ITEMS[0], aq10.ITEMS[1]
    assert aq10.item_point(item1, "definitely_agree") == 1
    assert aq10.item_point(item1, "slightly_disagree") == 0
    assert aq10.item_point(item2, "definitely_disagree") == 1
    assert aq10.item_point(item2, "slightly_agree") == 0


@pytest.mark.parametrize("n", range(11))
def test_total_score(n):
    assert aq10.score(answers_scoring(n)).score == n


def test_threshold():
    assert not aq10.score(answers_scoring(5)).above_threshold
    assert aq10.score(answers_scoring(6)).above_threshold


def test_incomplete_raises():
    with pytest.raises(ValueError):
        aq10.score({"q1": "definitely_agree"})


def test_pages_render(client):
    for path in ("/", "/screen", "/about"):
        assert client.get(path).status_code == 200


def test_result_above(client):
    r = client.post("/result", data=answers_scoring(8))
    assert r.status_code == 200
    assert b"full assessment could be worth asking for" in r.data


def test_result_below(client):
    r = client.post("/result", data=answers_scoring(2))
    assert b"below the usual referral level" in r.data


def test_missing_answers_returns_form(client):
    data = answers_scoring(3)
    del data["q4"]
    r = client.post("/result", data=data)
    assert r.status_code == 400
    assert b"Answer question 4" in r.data
