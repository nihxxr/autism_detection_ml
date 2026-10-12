from pathlib import Path

from flask import Flask, render_template, request, abort

from . import aq10
from .model import load_model

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "aq10_model.joblib"


def create_app() -> Flask:
    app = Flask(__name__)
    model = load_model(MODEL_PATH)

    @app.context_processor
    def inject_constants():
        return {"answers": aq10.ANSWERS, "items": aq10.ITEMS, "threshold": aq10.REFERRAL_THRESHOLD}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/screen")
    def screen():
        return render_template("screen.html", error=None, chosen={})

    @app.post("/result")
    def result():
        chosen = {k: v for k, v in request.form.items() if k.startswith("q")}
        try:
            res = aq10.score(chosen)
        except ValueError:
            missing = [i.number for i in aq10.ITEMS if f"q{i.number}" not in chosen]
            error = (
                f"Answer question {missing[0]} to see your result."
                if len(missing) == 1
                else f"Answer questions {', '.join(map(str, missing))} to see your result."
            )
            return render_template("screen.html", error=error, chosen=chosen), 400

        model_estimate = model.probability(res.points) if model else None
        rows = list(zip(aq10.ITEMS, res.answers, res.points))
        return render_template("result.html", res=res, rows=rows, model_estimate=model_estimate)

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.errorhandler(404)
    def not_found(_):
        return render_template("index.html"), 404

    return app
