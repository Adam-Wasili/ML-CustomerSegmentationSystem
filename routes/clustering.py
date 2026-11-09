"""ML clustering routes (Module 4)."""
import pandas as pd
from flask import (Blueprint, abort, current_app, flash, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_required
from sqlalchemy import select, update

from models import AnalysisRun, Dataset, RFMResult, Segment, db
from services.behavioral_analytics import BehavioralAnalytics
from services.clustering import MLClustering, MLClusteringError

clustering_bp = Blueprint("clustering", __name__, url_prefix="/clustering")

GENERIC_ERROR = "Something went wrong while clustering the customers. Please try again."


# ------------------------------------------------------------------ helpers
def _get_owned_run(run_id):
    """Return the analysis run only if its dataset belongs to the logged-in user."""
    run = db.session.get(AnalysisRun, run_id)
    if run is None or run.dataset.user_id != current_user.id:
        abort(404)
    return run


def _load_rfm(run):
    """The RFM results of a run as a DataFrame (the order of the rows is fixed by id)."""
    columns = ["id", "recency", "frequency", "monetary", "r_score", "f_score", "m_score",
               "cluster_label"]
    rows = db.session.execute(
        select(RFMResult.id, RFMResult.recency, RFMResult.frequency, RFMResult.monetary,
               RFMResult.r_score, RFMResult.f_score, RFMResult.m_score, RFMResult.cluster_label)
        .where(RFMResult.run_id == run.id)
        .order_by(RFMResult.id)
    ).all()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


def _saved_results(run, rfm):
    """The saved clusters of a run as a list of rows for the page (or None if not clustered yet)."""
    if run.n_clusters is None:
        return None
    labelled = rfm.dropna(subset=["cluster_label"])
    if labelled.empty:
        return None
    profile = MLClustering.profile_clusters(labelled, labelled["cluster_label"].astype(int))
    segments = {segment.cluster_label: segment
                for segment in Segment.query.filter_by(run_id=run.id)}
    rows = []
    for row in profile.itertuples(index=False):
        segment = segments.get(int(row.cluster))
        rows.append({
            "cluster": int(row.cluster),
            "name": segment.segment_name if segment else f"Cluster {int(row.cluster)}",
            "description": segment.description if segment else "",
            "customers": int(row.customers),
            "percent": float(row.percent),
            "avg_recency": float(row.avg_recency),
            "avg_frequency": float(row.avg_frequency),
            "avg_monetary": float(row.avg_monetary),
        })
    return rows


def _save_clusters(run, rfm, outcome, profile):
    """Store the segments and the cluster of every customer (replacing any earlier result)."""
    # Clear the earlier result of this run first
    RFMResult.query.filter_by(run_id=run.id).update({"cluster_label": None, "segment_id": None})
    Segment.query.filter_by(run_id=run.id).delete()

    # One Segment row per cluster, with the name and description chosen from its statistics
    segment_ids = {}
    for row in profile.itertuples(index=False):
        segment = Segment(run_id=run.id, cluster_label=int(row.cluster),
                          segment_name=row.segment_name, description=row.description)
        db.session.add(segment)
        db.session.flush()                          # gives the segment its id
        segment_ids[int(row.cluster)] = segment.id

    # Give every customer's RFM row its cluster and segment (one bulk update)
    updates = [
        {"id": int(row_id), "cluster_label": int(label), "segment_id": segment_ids[int(label)]}
        for row_id, label in zip(rfm["id"], outcome["labels"])
    ]
    db.session.execute(update(RFMResult), updates)

    run.n_clusters = outcome["k"]
    run.silhouette = outcome["silhouette"]
    run.inertia = outcome["inertia"]


# ------------------------------------------------------------------- routes
@clustering_bp.route("/")
@login_required
def index():
    """Sidebar link: open the clustering page of the user's latest RFM analysis."""
    latest = (AnalysisRun.query
              .join(Dataset, AnalysisRun.dataset_id == Dataset.id)
              .filter(Dataset.user_id == current_user.id)
              .order_by(AnalysisRun.id.desc()).first())
    if latest is None:
        flash("Calculate the RFM analysis first, then continue to ML Clustering.", "info")
        return redirect(url_for("rfm.index"))
    return redirect(url_for("clustering.show_run", run_id=latest.id))


@clustering_bp.route("/run/<int:run_id>")
@login_required
def show_run(run_id):
    """The clustering page: preparation, choice of k, and the saved result."""
    run = _get_owned_run(run_id)
    rfm = _load_rfm(run)

    evaluation = recommended = error = None
    try:
        clustering = MLClustering()
        features = clustering.prepare_features(rfm)
        evaluation = clustering.evaluate_k(features)       # tries k = 2 ... 10
        recommended = clustering.recommend_k(evaluation)
    except MLClusteringError as problem:                   # for example: too few customers
        error = str(problem)

    return render_template("clustering.html", run=run, evaluation=evaluation,
                           recommended=recommended, error=error,
                           selected_k=run.n_clusters or recommended,
                           results=_saved_results(run, rfm))


@clustering_bp.route("/run/<int:run_id>/cluster", methods=["POST"])
@login_required
def cluster(run_id):
    """Run K-Means with the number of clusters chosen by the user."""
    run = _get_owned_run(run_id)
    try:
        k = int(request.form.get("n_clusters", ""))
    except ValueError:
        flash("Choose the number of clusters from the list.", "danger")
        return redirect(url_for("clustering.show_run", run_id=run.id))

    try:
        rfm = _load_rfm(run)
        clustering = MLClustering()
        features = clustering.prepare_features(rfm)
        outcome = clustering.perform_clustering(features, k)
        profile = BehavioralAnalytics().identify_characteristics(
            clustering.profile_clusters(rfm, outcome["labels"]))
        _save_clusters(run, rfm, outcome, profile)
        db.session.commit()
    except MLClusteringError as problem:                   # problems we can explain to the user
        db.session.rollback()
        flash(str(problem), "danger")
        return redirect(url_for("clustering.show_run", run_id=run.id))
    except Exception:                                      # anything unexpected: log it, show a safe message
        db.session.rollback()
        current_app.logger.exception("Clustering failed")
        flash(GENERIC_ERROR, "danger")
        return redirect(url_for("clustering.show_run", run_id=run.id))

    flash(f"The customers were grouped into {k} clusters.", "success")
    return redirect(url_for("clustering.show_run", run_id=run.id))