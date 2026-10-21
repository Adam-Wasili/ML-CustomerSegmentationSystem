"""Database tables for the whole project (all modules).

Each class below is one table. SQLAlchemy turns these classes into SQL tables,
so we work with Python objects instead of writing SQL by hand.
"""
from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

# The database object shared by the whole application (connected in app.py)
db = SQLAlchemy()


def _utcnow():
    """Current time in UTC, used as the default for 'created' columns."""
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    """A person who can log in. Every user has the same access (no roles)."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)   # never the plain password
    created_at = db.Column(db.DateTime, default=_utcnow)

    # A user owns many datasets (user.datasets, dataset.owner)
    datasets = db.relationship("Dataset", backref="owner", lazy=True)

    def set_password(self, password):
        """Store a salted hash of the password. The original is never saved."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Compare a typed password with the stored hash."""
        return check_password_hash(self.password_hash, password)


class Dataset(db.Model):
    """One uploaded transaction file and the result of cleaning it."""
    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)    # original name, shown to the user
    stored_name = db.Column(db.String(255), nullable=False)  # random name used on disk
    file_type = db.Column(db.String(10), nullable=False)     # csv or xlsx
    upload_date = db.Column(db.DateTime, default=_utcnow)
    status = db.Column(db.String(20), default="uploaded")    # uploaded / cleaned
    rows_raw = db.Column(db.Integer, default=0)              # rows in the original file
    rows_retained = db.Column(db.Integer, default=0)         # rows kept after cleaning
    cleaning_summary = db.Column(db.Text)                    # JSON text: what cleaning did

    # Deleting a dataset also deletes its customers, transactions and analysis runs
    customers = db.relationship("Customer", backref="dataset", lazy=True,
                                cascade="all, delete-orphan")
    transactions = db.relationship("Transaction", backref="dataset", lazy=True,
                                   cascade="all, delete-orphan")
    runs = db.relationship("AnalysisRun", backref="dataset", lazy=True,
                           cascade="all, delete-orphan")


class Customer(db.Model):
    """One customer inside one dataset, taken from the Customer_ID in the file."""
    __tablename__ = "customers"
    # The same customer code cannot appear twice in the same dataset
    __table_args__ = (db.UniqueConstraint("dataset_id", "customer_code"),)

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False)
    customer_code = db.Column(db.String(100), nullable=False)  # the Customer_ID from the file
    customer_name = db.Column(db.String(200))                  # optional

    transactions = db.relationship("Transaction", backref="customer", lazy=True)


class Transaction(db.Model):
    """One purchase. Many transactions can belong to the same customer."""
    __tablename__ = "transactions"

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False, index=True)
    customer_id = db.Column(db.Integer, db.ForeignKey("customers.id"), nullable=False, index=True)
    transaction_date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    product = db.Column(db.String(200))     # optional
    category = db.Column(db.String(200))    # optional
    location = db.Column(db.String(200))    # optional


class AnalysisRun(db.Model):
    """One execution of RFM + clustering on a dataset.

    Keeping runs separate means the analysis can be repeated (for example with a
    different reference date or number of clusters) without losing earlier results.
    """
    __tablename__ = "analysis_runs"

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False)
    reference_date = db.Column(db.Date, nullable=False)   # date Recency is measured from
    n_clusters = db.Column(db.Integer)                    # k chosen for K-Means (filled in Step 4)
    silhouette = db.Column(db.Float)                      # clustering quality score
    inertia = db.Column(db.Float)                         # K-Means inertia (used by the elbow method)
    created_at = db.Column(db.DateTime, default=_utcnow)

    segments = db.relationship("Segment", backref="run", lazy=True,
                               cascade="all, delete-orphan")
    rfm_results = db.relationship("RFMResult", backref="run", lazy=True,
                                  cascade="all, delete-orphan")


class Segment(db.Model):
    """A customer group found by clustering, with a business name given afterwards."""
    __tablename__ = "segments"

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey("analysis_runs.id"), nullable=False)
    cluster_label = db.Column(db.Integer, nullable=False)   # 0, 1, 2 ... produced by K-Means
    segment_name = db.Column(db.String(100))                # for example "Loyal Customers"
    description = db.Column(db.Text)


class RFMResult(db.Model):
    """Recency, Frequency and Monetary values of one customer in one analysis run."""
    __tablename__ = "rfm_results"

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey("analysis_runs.id"), nullable=False, index=True)
    customer_id = db.Column(db.Integer, db.ForeignKey("customers.id"), nullable=False)
    recency = db.Column(db.Integer, nullable=False)     # days since the last transaction
    frequency = db.Column(db.Integer, nullable=False)   # number of transactions
    monetary = db.Column(db.Float, nullable=False)      # total amount spent
    r_score = db.Column(db.Integer)                     # 1-5 scores
    f_score = db.Column(db.Integer)
    m_score = db.Column(db.Integer)
    cluster_label = db.Column(db.Integer)               # empty until clustering is run
    segment_id = db.Column(db.Integer, db.ForeignKey("segments.id"))

    customer = db.relationship("Customer")