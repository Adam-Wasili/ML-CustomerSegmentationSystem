"""Tests for the clustering routes."""
import io

import pytest
from flask import g

from app import create_app
from config import TestingConfig
from models import AnalysisRun, Dataset, RFMResult, Segment, db

# Too few customers for clustering (only three)
CSV_SMALL = (
    "Customer_ID,Transaction_Date,Transaction_Amount\n"
    "C001,2025-01-05,25000\n"
    "C002,2025-01-07,15000\n"
    "C003,2025-02-10,20000\n"
)


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
    class ClusteringTestConfig(TestingConfig):
        UPLOAD_FOLDER = str(tmp_path)

    app = create_app(ClusteringTestConfig)
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


def prepare_run(client, content):
    """Upload a file and click the RFM button. Returns the id of the RFM run."""
    client.post("/datasets/upload",
                data={"file": (io.BytesIO(content.encode()), "sales.csv")},
                content_type="multipart/form-data")
    dataset_id = Dataset.query.order_by(Dataset.id.desc()).first().id
    client.post(f"/rfm/dataset/{dataset_id}/run", data={"reference_date": ""})
    return AnalysisRun.query.order_by(AnalysisRun.id.desc()).first().id


def cluster(client, run_id, k):
    return client.post(f"/clustering/run/{run_id}/cluster", data={"n_clusters": str(k)})


# ----------------------------------------------------------------- the page
def test_page_shows_the_evaluation_and_a_recommendation(client):
    run_id = prepare_run(client, three_group_csv())
    page = client.get(f"/clustering/run/{run_id}")
    assert page.status_code == 200
    assert b"Recommended number of clusters" in page.data
    assert b"Silhouette" in page.data


def test_too_few_customers_shows_a_message(client):
    run_id = prepare_run(client, CSV_SMALL)
    page = client.get(f"/clustering/run/{run_id}")
    assert page.status_code == 200
    assert b"At least 10 customers" in page.data


def test_nothing_can_be_downloaded(client):
    run_id = prepare_run(client, three_group_csv())
    assert b"Download" not in client.get(f"/clustering/run/{run_id}").data


# ------------------------------------------------------------- clustering
def test_clustering_saves_segments_and_labels(client):
    run_id = prepare_run(client, three_group_csv())
    response = cluster(client, run_id, 3)
    assert response.status_code == 302

    run = db.session.get(AnalysisRun, run_id)
    assert run.n_clusters == 3
    assert run.silhouette is not None and run.inertia is not None
    assert Segment.query.filter_by(run_id=run_id).count() == 3
    results = RFMResult.query.filter_by(run_id=run_id).all()
    assert len(results) == 30
    assert all(row.cluster_label is not None and row.segment_id is not None for row in results)


def test_result_page_shows_names_and_statistics(client):
    run_id = prepare_run(client, three_group_csv())
    cluster(client, run_id, 3)
    page = client.get(f"/clustering/run/{run_id}")
    assert b"Result: 3 clusters" in page.data
    assert b"Customers" in page.data
    assert b"do not predict" in page.data


def test_running_again_replaces_the_earlier_clusters(client):
    run_id = prepare_run(client, three_group_csv())
    cluster(client, run_id, 3)
    cluster(client, run_id, 2)
    assert db.session.get(AnalysisRun, run_id).n_clusters == 2
    assert Segment.query.filter_by(run_id=run_id).count() == 2


@pytest.mark.parametrize("bad_value", ["1", "99", "abc", ""])
def test_invalid_number_of_clusters_changes_nothing(client, bad_value):
    run_id = prepare_run(client, three_group_csv())
    response = client.post(f"/clustering/run/{run_id}/cluster",
                           data={"n_clusters": bad_value}, follow_redirects=True)
    assert b"Choose the number of clusters" in response.data or b"Choose a number of clusters" in response.data
    assert db.session.get(AnalysisRun, run_id).n_clusters is None
    assert Segment.query.count() == 0


# ------------------------------------------------------------ navigation
def test_sidebar_link_needs_an_rfm_analysis_first(client):
    response = client.get("/clustering/")
    assert response.status_code == 302
    assert "/rfm/" in response.headers["Location"]


def test_sidebar_link_opens_the_latest_run(client):
    run_id = prepare_run(client, three_group_csv())
    response = client.get("/clustering/")
    assert response.status_code == 302
    assert f"/clustering/run/{run_id}" in response.headers["Location"]


# --------------------------------------------------- ownership and deleting
def test_other_user_cannot_open_or_change_my_clusters(app, client):
    run_id = prepare_run(client, three_group_csv())
    other = sign_in_as_another_user(app, "bob")
    assert other.get(f"/clustering/run/{run_id}").status_code == 404
    assert other.post(f"/clustering/run/{run_id}/cluster", data={"n_clusters": "3"}).status_code == 404


def test_deleting_a_dataset_removes_its_segments(client):
    run_id = prepare_run(client, three_group_csv())
    cluster(client, run_id, 3)
    dataset_id = Dataset.query.one().id
    client.post(f"/datasets/{dataset_id}/delete")
    assert Segment.query.count() == 0
    assert RFMResult.query.count() == 0