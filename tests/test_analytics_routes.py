"""Tests for the dashboard, analytics, segments, visualization and report pages."""
import io

import pytest
from flask import g

from app import create_app
from config import TestingConfig
from models import AnalysisRun, Customer, Dataset, RFMResult, Segment, db


def three_group_csv():
    """30 customers: 10 frequent and recent, 10 occasional, 10 who bought once long ago."""
    lines = ["Customer_ID,Transaction_Date,Transaction_Amount"]
    for i in range(10):
        for day in range(8):
            lines.append(f"A{i:02d},2025-06-{10 + day:02d},{50000 + i * 1000}")
    for i in range(10):
        for day in range(2):
            lines.append(f"B{i:02d},2025-04-{10 + day:02d},{10000 + i * 500}")
    for i in range(10):
        lines.append(f"C{i:02d},2025-01-{10 + i:02d},{2000 + i * 100}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def app(tmp_path):
    """A test app that stores uploaded files in a temporary folder."""
    class AnalyticsTestConfig(TestingConfig):
        UPLOAD_FOLDER = str(tmp_path)

    app = create_app(AnalyticsTestConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def sign_in(client, username):
    client.post("/register", data={"username": username, "password": "strongpass1",
                                   "confirm_password": "strongpass1"})
    client.post("/login", data={"username": username, "password": "strongpass1"})


def sign_in_as_another_user(app, username):
    """A second client logged in as a DIFFERENT user (see the note in the other test files)."""
    g.pop("_login_user", None)
    other = app.test_client()
    sign_in(other, username)
    return other


@pytest.fixture
def client(app):
    client = app.test_client()
    sign_in(client, "adam")
    return client


def upload(client):
    client.post("/datasets/upload",
                data={"file": (io.BytesIO(three_group_csv().encode()), "sales.csv")},
                content_type="multipart/form-data")


def calculate_rfm(client):
    dataset_id = Dataset.query.order_by(Dataset.id.desc()).first().id
    client.post(f"/rfm/dataset/{dataset_id}/run", data={"reference_date": ""})
    return AnalysisRun.query.order_by(AnalysisRun.id.desc()).first().id


def clustered_run(client):
    """Upload, calculate RFM and run K-Means with 3 clusters. Returns the run id."""
    upload(client)
    run_id = calculate_rfm(client)
    client.post(f"/clustering/run/{run_id}/cluster", data={"n_clusters": "3"})
    return run_id


def cluster_of(run_id, customer_code):
    return (RFMResult.query
            .join(Customer, RFMResult.customer_id == Customer.id)
            .filter(RFMResult.run_id == run_id, Customer.customer_code == customer_code)
            .one().cluster_label)


PAGES = ["/analytics/run/{id}", "/segments/run/{id}", "/visualizations/run/{id}", "/reports/run/{id}"]


# ------------------------------------------------------------------ the pages
@pytest.mark.parametrize("path", PAGES)
def test_every_page_opens_after_clustering(client, path):
    run_id = clustered_run(client)
    assert client.get(path.format(id=run_id)).status_code == 200


@pytest.mark.parametrize("path", PAGES)
def test_pages_need_clustering_first(client, path):
    upload(client)
    run_id = calculate_rfm(client)                       # RFM only, no clustering yet
    response = client.get(path.format(id=run_id))
    assert response.status_code == 302
    assert f"/clustering/run/{run_id}" in response.headers["Location"]


@pytest.mark.parametrize("path", ["/analytics/", "/segments/", "/visualizations/", "/reports/"])
def test_sidebar_links_without_clustering_go_to_ml_clustering(client, path):
    response = client.get(path)
    assert response.status_code == 302
    assert "/clustering/" in response.headers["Location"]


@pytest.mark.parametrize("path, target", [("/analytics/", "/analytics/run/"),
                                          ("/segments/", "/segments/run/"),
                                          ("/visualizations/", "/visualizations/run/"),
                                          ("/reports/", "/reports/run/")])
def test_sidebar_links_open_the_latest_clustered_run(client, path, target):
    clustered_run(client)
    response = client.get(path)
    assert response.status_code == 302
    assert target in response.headers["Location"]


def test_analytics_page_shows_revenue_share_and_the_no_prediction_note(client):
    run_id = clustered_run(client)
    page = client.get(f"/analytics/run/{run_id}")
    assert b"Revenue share" in page.data
    assert b"does not" in page.data and b"predict" in page.data
    assert b"Key observations" in page.data


def test_report_covers_the_whole_analysis(client):
    run_id = clustered_run(client)
    page = client.get(f"/reports/run/{run_id}")
    assert b"Data and cleaning" in page.data
    assert b"RFM analysis" in page.data
    assert b"silhouette" in page.data
    assert b"Customer segments" in page.data


def test_visualization_page_has_the_charts(client):
    run_id = clustered_run(client)
    page = client.get(f"/visualizations/run/{run_id}")
    assert b"pcaChart" in page.data
    assert b"PCA is used only" in page.data


# ---------------------------------------------------------- customer segments
def test_segment_filter_shows_only_that_segments_customers(client):
    run_id = clustered_run(client)
    label = cluster_of(run_id, "A00")
    page = client.get(f"/segments/run/{run_id}?segment={label}")
    assert page.status_code == 200
    assert b"A00" in page.data
    assert b"C00" not in page.data                       # the long-ago buyers are in another segment


def test_unknown_segment_filter_is_ignored(client):
    run_id = clustered_run(client)
    page = client.get(f"/segments/run/{run_id}?segment=999&page=abc")
    assert page.status_code == 200
    assert b"A00" in page.data and b"C09" in page.data


# ------------------------------------------------------------------ dashboard
def test_dashboard_for_a_new_user_invites_an_upload(client):
    page = client.get("/dashboard")
    assert page.status_code == 200
    assert b"No dataset has been uploaded yet" in page.data


def test_dashboard_follows_the_progress_of_the_analysis(client):
    upload(client)
    assert b"Total customers" in client.get("/dashboard").data
    assert b"Calculate RFM Analysis" in client.get("/dashboard").data
    run_id = calculate_rfm(client)
    assert b"Go to ML Clustering" in client.get("/dashboard").data
    client.post(f"/clustering/run/{run_id}/cluster", data={"n_clusters": "3"})
    page = client.get("/dashboard")
    assert b"Segment distribution" in page.data
    assert b"Number of segments" in page.data


# -------------------------------------------------- ownership, no downloads
def test_other_user_cannot_open_my_analysis_pages(app, client):
    run_id = clustered_run(client)
    other = sign_in_as_another_user(app, "bob")
    for path in PAGES:
        assert other.get(path.format(id=run_id)).status_code == 404


def test_nothing_can_be_downloaded(client):
    run_id = clustered_run(client)
    for path in PAGES + ["/dashboard"]:
        assert b"Download" not in client.get(path.format(id=run_id)).data