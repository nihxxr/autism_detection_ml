# AQ-10 Screener

A small Flask app that walks adults through the AQ-10, a ten question autism screening questionnaire, and explains the score in plain language.

You answer one statement per screen. The result page shows your score out of 10, where it sits against the referral level used in NICE guidance, and which statements scored. You can print it or save it as a PDF to take to a clinician.

The app screens. It does not diagnose.

## Why this replaced the original ML version

The first version trained a random forest on the UCI Autism Screening Adult dataset and called the output a prediction. Two problems came up when I rebuilt it:

1. The UCI label is computed from the AQ-10 total. Every row with a score above 6 is marked positive. A model trained on the ten answers just relearns that sum, so a near perfect accuracy number means nothing.
2. The questions on the old form were paraphrases, not the validated AQ-10 items, and the answers were typed as 0 or 1.

This version uses the published items and scoring rules directly. `scripts/train_model.py` still runs the experiment and prints the plain sum rule next to logistic regression and a random forest, so you can see the leak in the numbers. If you run it, the result page also shows the trained model's probability as a cross check.

## Project layout

```
app/
  __init__.py       Flask app factory and routes
  aq10.py           Questionnaire items, answer scale, scoring
  model.py          Optional loader for a trained model
  templates/        Jinja templates
  static/           CSS, JS, self hosted Atkinson Hyperlegible font
scripts/
  train_model.py    Downloads UCI dataset 426, evaluates models, saves one
tests/
  test_app.py       Scoring and route tests
wsgi.py             Entry point for gunicorn and local runs
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python wsgi.py                   # http://localhost:5000
pytest                           # 18 tests
```

To train the optional model:

```bash
python scripts/train_model.py              # fetches the UCI data
python scripts/train_model.py path/to.csv  # or use a local CSV or ARFF file
```

This writes `models/aq10_model.joblib` and `models/metrics.json`.

## Deploy

The `Procfile` runs `gunicorn wsgi:app`, so Render, Railway or Heroku pick it up without extra config. No environment variables are needed.

## Design notes

The audience includes people who find busy, bright or animated pages hard to use, so the interface stays calm on purpose:

- One typeface, Atkinson Hyperlegible, designed by the Braille Institute for low vision readers. It's self hosted, so the app makes no third party requests.
- Muted grey green background and a single teal for actions. No animation anywhere.
- One question per screen with large tap targets. Keyboard works throughout: arrow keys pick an answer, Enter moves on.
- Without JavaScript the form still works and shows every question on one page.
- Nothing you answer is stored. The result page is rendered from the form post and never saved.

## Sources

- Allison C, Auyeung B, Baron-Cohen S (2012). Toward brief "red flags" for autism screening. *Journal of the American Academy of Child and Adolescent Psychiatry*, 51(2), 202 to 212.
- NICE guideline CG142: Autism spectrum disorder in adults: diagnosis and management.
- Thabtah F. Autism Screening Adult dataset, UCI Machine Learning Repository, id 426.
