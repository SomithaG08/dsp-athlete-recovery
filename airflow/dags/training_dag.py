"""Training job: retrain on newly validated data, promote only if better.

load_data (skips if not enough new data) -> train_model -> save_training_stats -> evaluate_candidate
-> promote_to_champion (branch)
       promoted     -> notify_api_reload (+ "promoted" card, Teams ML channel) + archive_data
       not promoted -> send_ml_alert ("not promoted" card, Teams ML channel)

Step logic lives in scripts/training_pipeline.py; each task passes a small dict through XCom
(file paths, run id, version, metrics), never the data itself.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, task

PROJECT_DIR = Path(os.getenv("DSP_PROJECT_DIR", "/opt/airflow/project"))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
SCHEDULE = os.getenv("TRAINING_SCHEDULE", "0 * * * *")


def _ml_alert_facts(evaluation):
    """Facts shown on both ML cards (promoted / not promoted)."""
    cand = evaluation["candidate_metrics"]
    champ = evaluation.get("champion_metrics") or {}
    champion = (f"v{evaluation['champion_version']} RMSE {champ['rmse']:.3f}"
                if evaluation.get("champion_version") and "rmse" in champ else "none")
    return {
        "Reason": evaluation["reason"],
        "Candidate RMSE": f"{cand['rmse']:.3f}",
        "Previous champion": champion,
        "Inference": f"{evaluation['inference_ms_per_row']:.3f} ms/row",
        "Training rows": evaluation["train_rows"],
    }


@dag(schedule=SCHEDULE, start_date=datetime(2026, 1, 1), catchup=False, max_active_runs=1,
     tags=["athlete", "training", "mlflow"])
def athlete_model_retraining():

    @task
    def load_data():
        """Check that enough new rows reached good_data since the last training; if so, load them."""
        from scripts.training_pipeline import check_for_new_data as check
        from scripts.training_pipeline import load_data as load

        plan = check()
        print(plan["reason"])
        if not plan["ready"]:
            raise AirflowSkipException(plan["reason"])
        return load(plan)

    @task
    def train_model(data):
        from scripts.training_pipeline import train_model as train
        result = train(data)
        print(f"Candidate v{result['version']} holdout metrics: {result['metrics']}")
        return result

    @task
    def save_training_stats(candidate):
        from scripts.training_pipeline import save_training_stats as save
        return save(candidate)

    @task
    def evaluate_candidate(candidate):
        from scripts.training_pipeline import evaluate_candidate as evaluate
        result = evaluate(candidate)
        print(f"Decision: {'promote' if result['promote'] else 'reject'} — {result['reason']}")
        return result

    @task.branch
    def promote_to_champion(evaluation):
        from scripts.training_pipeline import promote_to_champion as promote
        promote(evaluation)
        return ["notify_api_reload", "archive_data"] if evaluation["promote"] else ["send_ml_alert"]

    @task
    def notify_api_reload(evaluation):
        from scripts.teams import send_alert

        api_url = os.getenv("MODEL_SERVICE_URL", "http://fastapi:8000")
        response = requests.post(f"{api_url}/reload-model", timeout=60)
        response.raise_for_status()
        print(f"FastAPI reloaded production model: {response.json()}")
        # Sent only after the API serves the new model, so the card means "live in production".
        send_alert(
            "ml",
            f"Model v{evaluation['version']} promoted to champion",
            _ml_alert_facts(evaluation),
            link=evaluation["run_url"],
            link_title="Open MLflow run",
            severity="good",
        )

    @task
    def archive_data(evaluation):
        from scripts.training_pipeline import archive_data as archive
        moved = archive(evaluation)
        print(f"Archived {len(moved)} used training file(s)")

    @task
    def send_ml_alert(evaluation):
        from scripts.teams import send_alert
        send_alert(
            "ml",
            f"Model v{evaluation['version']} not promoted",
            _ml_alert_facts(evaluation),
            link=evaluation["run_url"],
            link_title="Open MLflow run",
        )

    evaluation = evaluate_candidate(save_training_stats(train_model(load_data())))
    branch = promote_to_champion(evaluation)
    branch >> [notify_api_reload(evaluation), archive_data(evaluation), send_ml_alert(evaluation)]


athlete_model_retraining()
