"""Upload, preview and results routes (Module 2).

Flow: the user uploads a file -> it is checked, cleaned and saved automatically
-> the user lands on the results page showing what happened to the data.
"""
import json
import os
import uuid

from flask import (Blueprint, abort, current_app, flash, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_required
from sqlalchemy import insert, select
from werkzeug.utils import secure_filename

from models import (AnalysisRun, Customer, Dataset, RFMResult, Segment,
                    Transaction, db)
from services.data_cleaning import (REMOVAL_LABELS, REQUIRED_COLUMNS,
                                    DataProcessor, DatasetError)

upload_bp = Blueprint("upload", __name__, url_prefix="/datasets")

# Shown instead of technical details when something unexpected fails
GENERIC_ERROR = "Something went wrong while processing the file. Please try again."


# ------------------------------------------------------------------ helpers
def _extension_allowed(filename):
    """Only extensions listed in config (csv, xlsx) are accepted."""
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"])


def _content_matches_extension(path, extension):
    """Do not trust the file name: look at the first bytes of the file as well."""
    with open(path, "rb") as handle:
        head = handle.read(4096)
    if extension == "xlsx":
        return head.startswith(b"PK\x03\x04")      # .xlsx files are zip archives
    return b"\x00" not in head                      # a .csv must be text, not binary


def _dataset_path(dataset):
    return os.path.join(current_app.config["UPLOAD_FOLDER"], dataset.stored_name)


def _get_owned_dataset(dataset_id):
    """Return the dataset only if it belongs to the logged-in user (otherwise 404)."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None or dataset.user_id != current_user.id:
        abort(404)
    return dataset


def _remove_file(path):
    """Delete a file, ignoring the error if it is already gone."""
    try:
        os.remove(path)
    except OSError:
        pass


def _render_upload(status=200):
    """Show the upload page with the user's existing datasets."""
    datasets = (Dataset.query.filter_by(user_id=current_user.id)
                .order_by(Dataset.upload_date.desc()).all())
    return render_template("upload.html", datasets=datasets,
                           required_columns=REQUIRED_COLUMNS), status


def _save_clean_data(dataset, clean_df):
    """Store customers and transactions.

    Customers are created from the unique Customer_ID values in the file (the
    system never makes up IDs). Every transaction is linked to its customer, so
    three rows with Customer_ID 'C003' belong to ONE customer.
    """
    # First non-empty name found for each customer
    first_names = (clean_df.dropna(subset=["Customer_Name"])
                   .drop_duplicates("Customer_ID")
                   .set_index("Customer_ID")["Customer_Name"].to_dict())

    customer_rows = [
        {"dataset_id": dataset.id, "customer_code": code, "customer_name": first_names.get(code)}
        for code in clean_df["Customer_ID"].drop_duplicates()
    ]
    db.session.execute(insert(Customer), customer_rows)

    # Look up the database id of each customer from its code
    code_to_id = {
        code: customer_id for code, customer_id in db.session.execute(
            select(Customer.customer_code, Customer.id).where(Customer.dataset_id == dataset.id))
    }

    transaction_rows = [
        {
            "dataset_id": dataset.id,
            "customer_id": code_to_id[row.Customer_ID],   # same code -> same customer
            "transaction_date": row.Transaction_Date.date(),
            "amount": float(row.Transaction_Amount),
            "product": row.Product,
            "category": row.Category,
            "location": row.Location,
        }
        for row in clean_df.itertuples(index=False)
    ]
    db.session.execute(insert(Transaction), transaction_rows)


# ------------------------------------------------------------------- routes
@upload_bp.app_errorhandler(413)
def file_too_large(_error):
    """Flask raises 413 when a file is bigger than MAX_CONTENT_LENGTH (10 MB)."""
    if not current_user.is_authenticated:
        return redirect(url_for("auth.login"))
    flash("The file is too large. The maximum size is 10 MB.", "danger")
    return _render_upload(413)


@upload_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    """GET: show the upload page. POST: check, clean and save the file."""
    if request.method == "GET":
        return _render_upload()

    # ---- 1. basic checks on the file itself ----
    file = request.files.get("file")
    if file is None or file.filename == "":
        flash("Please choose a file to upload.", "danger")
        return _render_upload(400)

    original_name = secure_filename(file.filename)   # removes dangerous characters
    if not original_name or not _extension_allowed(original_name):
        flash("Only .csv and .xlsx files are allowed.", "danger")
        return _render_upload(400)

    extension = original_name.rsplit(".", 1)[1].lower()
    stored_name = f"{uuid.uuid4().hex}.{extension}"  # random name: no path tricks, no clashes
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], stored_name)
    file.save(path)

    if not _content_matches_extension(path, extension):
        _remove_file(path)
        flash("The file content does not match its extension.", "danger")
        return _render_upload(400)

    # ---- 2. read, validate and clean automatically ----
    try:
        raw_df = DataProcessor.read_file(path, extension)
        processor = DataProcessor()
        df = processor.normalize_columns(raw_df)

        structure_errors = processor.validate_structure(df)
        if structure_errors:
            # Required columns are missing: nothing is saved and the file is deleted
            _remove_file(path)
            for message in structure_errors:
                flash(message, "danger")
            return _render_upload(400)

        clean_df, summary = processor.clean_data(df)

        # ---- 3. save everything in ONE database transaction ----
        dataset = Dataset(
            user_id=current_user.id, file_name=original_name, stored_name=stored_name,
            file_type=extension, status="cleaned",
            rows_raw=summary["rows_raw"], rows_retained=summary["rows_retained"],
            cleaning_summary=json.dumps(summary),
        )
        db.session.add(dataset)
        db.session.flush()                 # gives the dataset its id without committing yet
        _save_clean_data(dataset, clean_df)
        db.session.commit()                # all or nothing: no half-saved datasets
    except DatasetError as error:          # problems we can explain to the user
        db.session.rollback()
        _remove_file(path)
        flash(str(error), "danger")
        return _render_upload(400)
    except Exception:                      # anything unexpected: log it, show a safe message
        db.session.rollback()
        _remove_file(path)
        current_app.logger.exception("Unexpected error while processing an upload")
        flash(GENERIC_ERROR, "danger")
        return _render_upload(500)

    flash("File uploaded, validated and cleaned successfully.", "success")
    return redirect(url_for("upload.results", dataset_id=dataset.id))


@upload_bp.route("/<int:dataset_id>/preview")
@login_required
def preview(dataset_id):
    """Show the first rows of the original file and a data quality check."""
    dataset = _get_owned_dataset(dataset_id)
    try:
        raw_df = DataProcessor.read_file(_dataset_path(dataset), dataset.file_type)
        processor = DataProcessor()
        df = processor.normalize_columns(raw_df)
        quality = processor.inspect_quality(df)   # counts problems, changes nothing
    except Exception:
        current_app.logger.exception("Could not build the dataset preview")
        flash(GENERIC_ERROR, "danger")
        return redirect(url_for("upload.upload"))

    return render_template(
        "preview.html", dataset=dataset,
        preview_columns=list(raw_df.columns),
        preview_rows=raw_df.head(20).values.tolist(),
        n_rows=len(raw_df), n_columns=len(raw_df.columns),
        renamed=processor.summary["columns_renamed"],
        ignored=processor.summary["columns_ignored"],
        quality=quality,
    )


@upload_bp.route("/<int:dataset_id>/results")
@login_required
def results(dataset_id):
    """Show the outcome of cleaning (rows removed, corrections, warnings)."""
    dataset = _get_owned_dataset(dataset_id)
    if not dataset.cleaning_summary:
        abort(404)
    # If RFM was already calculated, the page offers "View" instead of "Calculate"
    latest_run = (AnalysisRun.query.filter_by(dataset_id=dataset.id)
                  .order_by(AnalysisRun.id.desc()).first())
    return render_template("cleaning.html", dataset=dataset,
                           summary=json.loads(dataset.cleaning_summary),
                           removal_labels=REMOVAL_LABELS, latest_run=latest_run)


@upload_bp.route("/<int:dataset_id>/delete", methods=["POST"])
@login_required
def delete(dataset_id):
    """Delete a dataset together with everything that depends on it."""
    dataset = _get_owned_dataset(dataset_id)
    # Delete children first (RFM results and segments belong to analysis runs)
    run_ids = [run.id for run in AnalysisRun.query.filter_by(dataset_id=dataset.id)]
    if run_ids:
        RFMResult.query.filter(RFMResult.run_id.in_(run_ids)).delete(synchronize_session=False)
        Segment.query.filter(Segment.run_id.in_(run_ids)).delete(synchronize_session=False)
        AnalysisRun.query.filter(AnalysisRun.id.in_(run_ids)).delete(synchronize_session=False)
    Transaction.query.filter_by(dataset_id=dataset.id).delete(synchronize_session=False)
    Customer.query.filter_by(dataset_id=dataset.id).delete(synchronize_session=False)
    _remove_file(_dataset_path(dataset))
    db.session.delete(dataset)
    db.session.commit()
    flash("Dataset deleted.", "info")
    return redirect(url_for("upload.upload"))