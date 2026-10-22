"""RFM analysis routes (Module 3).

The RFM analysis is NOT started automatically. After cleaning, the user clicks
the RFM button. Recency is always measured against the dataset's own last
record, never against the date on which the analysis is run, because the
uploaded datasets are usually old.
"""
import json
from datetime import date, timedelta

import pandas as pd
from flask import (Blueprint, abort, current_app, flash, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_required
from sqlalchemy import func, insert, select
from sqlalchemy.orm import joinedload

from models import AnalysisRun, Dataset, RFMResult, Transaction, db
from services.data_cleaning import REMOVAL_LABELS
from services.rfm_analysis import RFMAnalysis, RFMError

rfm_bp = Blueprint("rfm", __name__, url_prefix="/rfm")

PAGE_SIZE = 25
GENERIC_ERROR = "Something went wrong while calculating the RFM analysis. Please try again."

# Columns the results table may be sorted by. Only these names are accepted
# from the address bar, so users cannot sort by anything else.
SORT_COLUMNS = {
    "recency": RFMResult.recency,
    "frequency": RFMResult.frequency,
    "monetary": RFMResult.monetary,
    "r_score": RFMResult.r_score,
    "f_score": RFMResult.f_score,
    "m_score": RFMResult.m_score,
}


# ------------------------------------------------------------------ helpers
def _get_owned_dataset(dataset_id):
    """Return the dataset only if it belongs to the logged-in user (otherwise 404)."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None or dataset.user_id != current_user.id:
        abort(404)
    return dataset


def _get_owned_run(run_id):
    """Return the analysis run only if its dataset belongs to the logged-in user."""
    run = db.session.get(AnalysisRun, run_id)
    if run is None or run.dataset.user_id != current_user.id:
        abort(404)
    return run


def _cleaning_summary(dataset):
    """The saved cleaning report of a dataset (a dict), or None."""
    return json.loads(dataset.cleaning_summary) if dataset.cleaning_summary else None


def _data_period(dataset):
    """The period the dataset covers, taken from the transactions themselves.

    Returns the first and last record dates, the number of days between them
    (inclusive) and the DEFAULT analysis date (last record + 1 day). The date on
    which the analysis is run plays no part in any of these values.
    """
    start, end = db.session.execute(
        select(func.min(Transaction.transaction_date), func.max(Transaction.transaction_date))
        .where(Transaction.dataset_id == dataset.id)
    ).one()
    if start is None or end is None:
        return None
    return {
        "start": start,
        "end": end,
        "days": (end - start).days + 1,
        "default_analysis_date": end + timedelta(days=1),
    }


def create_rfm_run(dataset, reference_date=None):
    """Calculate RFM for a dataset and save it as a new analysis run.

    The caller must commit the database session. reference_date is a date or
    None (None = last transaction in the data + 1 day). Raises RFMError for bad input.
    """
    rows = db.session.execute(
        select(Transaction.customer_id, Transaction.transaction_date, Transaction.amount)
        .where(Transaction.dataset_id == dataset.id)
    ).all()
    transactions = pd.DataFrame([tuple(row) for row in rows],
                                columns=["customer_id", "transaction_date", "amount"])
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    analysis = RFMAnalysis()
    analysis_date = analysis.resolve_reference_date(transactions, reference_date)
    rfm = analysis.score_rfm(analysis.calculate_rfm(transactions, analysis_date))

    run = AnalysisRun(dataset_id=dataset.id, reference_date=analysis_date.date())
    db.session.add(run)
    db.session.flush()                      # gives the run its id without committing yet

    records = [
        {
            "run_id": run.id,
            "customer_id": int(row.customer_id),
            "recency": int(row.recency),
            "frequency": int(row.frequency),
            "monetary": float(row.monetary),
            "r_score": int(row.r_score),
            "f_score": int(row.f_score),
            "m_score": int(row.m_score),
        }
        for row in rfm.itertuples(index=False)
    ]
    db.session.execute(insert(RFMResult), records)
    return run


def load_rfm_dataframe(run):
    """All RFM results of a run as a DataFrame (used for the summary numbers)."""
    columns = ["recency", "frequency", "monetary", "r_score", "f_score", "m_score"]
    rows = db.session.execute(
        select(RFMResult.recency, RFMResult.frequency, RFMResult.monetary,
               RFMResult.r_score, RFMResult.f_score, RFMResult.m_score)
        .where(RFMResult.run_id == run.id)
    ).all()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


# ------------------------------------------------------------------- routes
@rfm_bp.route("/")
@login_required
def index():
    """Sidebar link: open the RFM page of the user's most recent dataset."""
    latest_dataset = (Dataset.query.filter_by(user_id=current_user.id)
                      .order_by(Dataset.id.desc()).first())
    if latest_dataset is None:
        flash("Upload a dataset first, then continue to the RFM analysis.", "info")
        return redirect(url_for("upload.upload"))
    return redirect(url_for("rfm.dataset_rfm", dataset_id=latest_dataset.id))


@rfm_bp.route("/dataset/<int:dataset_id>")
@login_required
def dataset_rfm(dataset_id):
    """Open the latest RFM run of a dataset, or offer the button to calculate one."""
    dataset = _get_owned_dataset(dataset_id)
    latest = (AnalysisRun.query.filter_by(dataset_id=dataset.id)
              .order_by(AnalysisRun.id.desc()).first())
    if latest is not None:
        return redirect(url_for("rfm.show_run", run_id=latest.id))
    # Nothing calculated yet: show the page with the "Calculate RFM Analysis" button
    return render_template("rfm.html", run=None, dataset=dataset,
                           cleaning=_cleaning_summary(dataset),
                           removal_labels=REMOVAL_LABELS, runs=[],
                           period=_data_period(dataset))


@rfm_bp.route("/run/<int:run_id>")
@login_required
def show_run(run_id):
    """The RFM page: cleaning outcome, data period, summary, score counts and table."""
    run = _get_owned_run(run_id)
    dataset = run.dataset

    # Sorting and paging come from the address bar, so check them first
    sort = request.args.get("sort", "monetary")
    direction = request.args.get("dir", "desc")
    if sort not in SORT_COLUMNS:
        sort = "monetary"
    if direction not in ("asc", "desc"):
        direction = "desc"
    column = SORT_COLUMNS[sort]
    order = column.desc() if direction == "desc" else column.asc()

    statement = (select(RFMResult)
                 .where(RFMResult.run_id == run.id)
                 .options(joinedload(RFMResult.customer))   # loads Customer_ID in the same query
                 .order_by(order, RFMResult.id))
    pagination = db.paginate(statement, page=request.args.get("page", 1, type=int),
                             per_page=PAGE_SIZE, error_out=False)

    summary = RFMAnalysis().summarise(load_rfm_dataframe(run))
    runs = (AnalysisRun.query.filter_by(dataset_id=dataset.id)
            .order_by(AnalysisRun.id.desc()).all())

    return render_template("rfm.html", run=run, dataset=dataset, summary=summary,
                           pagination=pagination, sort=sort, direction=direction,
                           runs=runs, cleaning=_cleaning_summary(dataset),
                           removal_labels=REMOVAL_LABELS, period=_data_period(dataset))


@rfm_bp.route("/dataset/<int:dataset_id>/run", methods=["POST"])
@login_required
def run_analysis(dataset_id):
    """Calculate RFM when the user clicks the button (optionally from another analysis date)."""
    dataset = _get_owned_dataset(dataset_id)
    text = request.form.get("reference_date", "").strip()

    # An empty date box means "use the default": the last record + 1 day
    reference_date = None
    if text:
        try:
            reference_date = date.fromisoformat(text)
        except ValueError:
            flash("The analysis date must look like YYYY-MM-DD.", "danger")
            return redirect(url_for("rfm.dataset_rfm", dataset_id=dataset.id))

    try:
        run = create_rfm_run(dataset, reference_date)
        db.session.commit()
    except RFMError as error:               # problems we can explain to the user
        db.session.rollback()
        flash(str(error), "danger")
        return redirect(url_for("rfm.dataset_rfm", dataset_id=dataset.id))
    except Exception:                       # anything unexpected: log it, show a safe message
        db.session.rollback()
        current_app.logger.exception("RFM calculation failed")
        flash(GENERIC_ERROR, "danger")
        return redirect(url_for("rfm.dataset_rfm", dataset_id=dataset.id))

    flash("RFM analysis calculated.", "success")
    return redirect(url_for("rfm.show_run", run_id=run.id))