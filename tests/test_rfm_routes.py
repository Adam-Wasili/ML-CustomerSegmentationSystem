"""Tests for the RFM routes. RFM starts only when the user clicks the RFM button."""
import io
from datetime import date

import pytest

from app import create_app
from config import TestingConfig
from models import AnalysisRun, Customer, Dataset, RFMResult, db

# Three customers; C003 made three purchases (total 60,000).
# The records run from 2025-01-05 to 2025-03-01, so the default analysis date is 2025-03-02.
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
    class RFMTestConfig(TestingConfig):
        UPLOAD_FOLDER = str(tmp_path)

    app = create_app(RFMTestConfig)
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


def upload(client, content=CSV_OK, filename="sales.csv"):
    """Upload a CSV file (validation and cleaning happen automatically)."""
    return client.post("/datasets/upload",
                       data={"file": (io.BytesIO(content.encode()), filename)},
                       content_type="multipart/form-data")


def calculate_rfm(client, reference_date=""):
    """Click the 'Calculate RFM Analysis' button. An empty date means the default."""
    dataset_id = Dataset.query.order_by(Dataset.id.desc()).first().id
    return client.post(f"/rfm/dataset/{dataset_id}/run", data={"reference_date": reference_date})


def result_for(run_id, customer_code):
    """The RFM row of one customer in one run."""
    return (RFMResult.query
            .join(Customer, RFMResult.customer_id == Customer.id)
            .filter(RFMResult.run_id == run_id, Customer.customer_code == customer_code)
            .one())


# ------------------------------------------- RFM starts only on the click
def test_upload_does_not_start_rfm_by_itself(client):
    response = upload(client)
    assert response.status_code == 302
    assert "/results" in response.headers["Location"]       # the cleaning outcome comes first
    assert AnalysisRun.query.count() == 0
    assert RFMResult.query.count() == 0


def test_clicking_the_rfm_button_calculates_rfm(client):
    upload(client)
    response = calculate_rfm(client)
    assert response.status_code == 302
    assert "/rfm/run/" in response.headers["Location"]
    assert AnalysisRun.query.count() == 1
    assert RFMResult.query.count() == 3                     # one row per customer


def test_results_page_offers_the_button_then_a_link(client):
    upload(client)
    dataset_id = Dataset.query.one().id
    assert b"Calculate RFM Analysis" in client.get(f"/datasets/{dataset_id}/results").data
    calculate_rfm(client)
    assert b"View RFM Analysis" in client.get(f"/datasets/{dataset_id}/results").data


# ------------------------------------------------- the calculated values
def test_rfm_values_group_transactions_by_customer_id(client):
    upload(client)
    calculate_rfm(client)
    run = AnalysisRun.query.one()
    c003 = result_for(run.id, "C003")
    assert c003.frequency == 3
    assert c003.monetary == 60000
    assert c003.recency == 1                    # last purchase 2025-03-01, analysis date 2025-03-02


def test_default_analysis_date_comes_from_the_data_not_from_today(client):
    upload(client)
    calculate_rfm(client)
    run = AnalysisRun.query.one()
    assert run.reference_date == date(2025, 3, 2)           # last record + 1 day
    assert run.reference_date != date.today()


def test_rfm_page_shows_cleaning_outcome_and_customers(client):
    upload(client)
    page = client.get(calculate_rfm(client).headers["Location"])
    assert page.status_code == 200
    assert b"Data cleaning outcome" in page.data
    assert b"C003" in page.data


def test_rfm_page_shows_the_data_period_and_analysis_date(client):
    upload(client)
    page = client.get(calculate_rfm(client).headers["Location"])
    assert b"2025-01-05" in page.data           # first record
    assert b"2025-03-01" in page.data           # last record
    assert b"2025-03-02" in page.data           # analysis date


# ------------------------------------------------------ analysis date
def test_recalculate_with_a_custom_analysis_date(client):
    upload(client)
    calculate_rfm(client)
    response = calculate_rfm(client, "2025-03-10")
    assert response.status_code == 302
    assert AnalysisRun.query.count() == 2                   # the first run is kept
    newest = AnalysisRun.query.order_by(AnalysisRun.id.desc()).first()
    assert newest.reference_date == date(2025, 3, 10)
    assert result_for(newest.id, "C003").recency == 9


def test_analysis_date_before_last_transaction_is_rejected(client):
    upload(client)
    calculate_rfm(client)
    dataset_id = Dataset.query.one().id
    response = client.post(f"/rfm/dataset/{dataset_id}/run",
                           data={"reference_date": "2025-01-01"}, follow_redirects=True)
    assert b"cannot be earlier" in response.data
    assert AnalysisRun.query.count() == 1


def test_analysis_date_that_is_not_a_date_is_rejected(client):
    upload(client)
    calculate_rfm(client)
    calculate_rfm(client, "abc")
    assert AnalysisRun.query.count() == 1


# ------------------------------------------------------------ the page
def test_sidebar_link_opens_the_rfm_page_of_the_latest_dataset(client):
    upload(client)
    first = client.get("/rfm/")
    assert first.status_code == 302
    assert "/rfm/dataset/" in first.headers["Location"]
    # Nothing calculated yet: the page offers the button
    assert b"Calculate RFM Analysis" in client.get("/rfm/", follow_redirects=True).data
    # After the click the same link shows the results
    calculate_rfm(client)
    assert b"RFM table" in client.get("/rfm/", follow_redirects=True).data


def test_rfm_without_any_dataset_redirects_to_upload(client):
    response = client.get("/rfm/")
    assert response.status_code == 302
    assert "/datasets/upload" in response.headers["Location"]


def test_bad_sort_and_page_values_do_not_break_the_page(client):
    upload(client)
    location = calculate_rfm(client).headers["Location"]
    assert client.get(location + "?sort=evil&dir=sideways&page=abc").status_code == 200


# ------------------------------------------- nothing can be downloaded
def test_nothing_can_be_downloaded(client):
    upload(client)
    location = calculate_rfm(client).headers["Location"]
    page = client.get(location)
    assert b"Download" not in page.data                     # no download button
    run_id = AnalysisRun.query.one().id
    assert client.get(f"/rfm/run/{run_id}/download").status_code == 404   # no download route


# ---------------------------------------------------------- ownership
def test_other_user_cannot_open_or_change_my_analysis(app, client):
    upload(client)
    calculate_rfm(client)
    run_id = AnalysisRun.query.one().id
    dataset_id = Dataset.query.one().id

    other = app.test_client()
    sign_in(other, "bob")
    assert other.get(f"/rfm/run/{run_id}").status_code == 404
    assert other.get(f"/rfm/dataset/{dataset_id}").status_code == 404
    assert other.post(f"/rfm/dataset/{dataset_id}/run", data={}).status_code == 404