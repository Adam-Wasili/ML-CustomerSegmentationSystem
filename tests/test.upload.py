"""Tests for the upload routes. Upload now validates and cleans automatically."""
import io
import json

import pytest

from app import create_app
from config import TestingConfig
from models import Customer, Dataset, Transaction, db

# Three customers; C003 has three transactions (total 60,000)
CSV_OK = (
    "Customer_ID,Transaction_Date,Transaction_Amount\n"
    "C001,2025-01-05,25000\n"
    "C002,2025-01-07,15000\n"
    "C003,2025-02-10,20000\n"
    "C003,2025-02-15,15000\n"
    "C003,2025-03-01,25000\n"
)


@pytest.fixture
def app(tmp_path):
    """A test app that stores uploaded files in a temporary folder."""
    class UploadTestConfig(TestingConfig):
        UPLOAD_FOLDER = str(tmp_path)

    app = create_app(UploadTestConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def sign_in(client, username):
    """Register and log in a user."""
    client.post("/register", data={"username": username, "password": "strongpass1",
                                   "confirm_password": "strongpass1"})
    client.post("/login", data={"username": username, "password": "strongpass1"})


@pytest.fixture
def client(app):
    client = app.test_client()
    sign_in(client, "adam")
    return client


def upload(client, content, filename="sales.csv"):
    """Post a file to the upload route (content can be text or bytes)."""
    data = content.encode() if isinstance(content, str) else content
    return client.post("/datasets/upload",
                       data={"file": (io.BytesIO(data), filename)},
                       content_type="multipart/form-data")


def test_upload_requires_login(app):
    response = app.test_client().post("/datasets/upload")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_valid_csv_is_cleaned_automatically(client):
    response = upload(client, CSV_OK)
    # The user is sent straight to the results page: no extra button needed
    assert response.status_code == 302
    assert "/results" in response.headers["Location"]
    dataset = Dataset.query.one()
    assert dataset.status == "cleaned"
    assert dataset.rows_raw == 5 and dataset.rows_retained == 5
    assert Customer.query.count() == 3
    assert Transaction.query.count() == 5


def test_results_page_shows_the_outcome(client):
    response = upload(client, CSV_OK)
    page = client.get(response.headers["Location"])
    assert page.status_code == 200
    assert b"Rows removed" in page.data


def test_invalid_file_type_rejected(client):
    assert upload(client, "hello", "notes.txt").status_code == 400
    assert Dataset.query.count() == 0


def test_fake_xlsx_rejected(client):
    assert upload(client, b"this is not a real excel file", "fake.xlsx").status_code == 400
    assert Dataset.query.count() == 0


def test_missing_required_columns_reported_and_nothing_saved(client):
    response = upload(client, "Customer_ID,Amount\nC1,100\n")
    assert response.status_code == 400
    assert b"Transaction_Date" in response.data     # clear message for the user
    assert Dataset.query.count() == 0


def test_customer_grouping_by_customer_id(client):
    upload(client, CSV_OK)
    c003 = Customer.query.filter_by(customer_code="C003").one()
    assert len(c003.transactions) == 3
    assert sum(t.amount for t in c003.transactions) == 60000


def test_removed_rows_are_reported(client):
    content = (
        "Customer_ID,Transaction_Date,Transaction_Amount\n"
        "C001,2025-01-05,25000\n"
        ",2025-01-06,100\n"            # missing Customer_ID
        "C002,2025-01-07,abc\n"        # invalid amount
        "C003,2025-13-45,500\n"        # invalid date
        "C001,2025-01-05,25000\n"      # duplicate
        "C004,2025-01-08,12000\n"
    )
    upload(client, content)
    summary = json.loads(Dataset.query.one().cleaning_summary)
    assert summary["rows_raw"] == 6
    assert summary["removed"]["missing_customer_id"] == 1
    assert summary["removed"]["invalid_date"] == 1
    assert summary["removed"]["invalid_amount"] == 1
    assert summary["removed"]["duplicates"] == 1
    assert summary["rows_retained"] == 2


def test_file_with_no_valid_rows_is_rejected_and_nothing_saved(client):
    response = upload(client, "Customer_ID,Transaction_Date,Transaction_Amount\n,,\n")
    assert response.status_code == 400
    assert Dataset.query.count() == 0


def test_user_cannot_open_another_users_dataset(app, client):
    upload(client, CSV_OK)
    dataset_id = Dataset.query.one().id
    other = app.test_client()
    sign_in(other, "bob")
    assert other.get(f"/datasets/{dataset_id}/results").status_code == 404
    assert other.get(f"/datasets/{dataset_id}/preview").status_code == 404