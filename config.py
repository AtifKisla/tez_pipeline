from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = OUTPUT_DIR / "models"

CITIES = [
    "Paris",
    "Rome",
    "Barcelona",
    "Amsterdam",
    "Prague",
    "Vienna",
    "Budapest",
    "Lisbon",
    "Berlin",
    "Athens",
    "London",
    "Madrid",
    "Istanbul",
    "Copenhagen",
    "Stockholm",
    "Dublin",
    "Brussels",
    "Warsaw",
    "Florence",
    "Venice",
]

ANCHOR_CITY = "Paris"

GOOGLE_TRENDS_START = "2018-01-01"
GOOGLE_TRENDS_END = "2025-12-31"
GOOGLE_TRENDS_RAW_PATH = RAW_DIR / "google_trends_monthly.csv"
GOOGLE_TRENDS_WEEKLY_PATH = INTERIM_DIR / "google_trends_weekly_wide.csv"
GOOGLE_TRENDS_MONTHLY_PATH = INTERIM_DIR / "google_trends_monthly.csv"

REDDIT_START = "2018-01-01"
REDDIT_END = "2025-12-31"
REDDIT_SOURCE_MODE = "dump"
REDDIT_RAW_PATH = RAW_DIR / "reddit_posts.csv"
REDDIT_STATUS_PATH = RAW_DIR / "reddit_collection_status.json"
REDDIT_CITY_AUDIT_PATH = RAW_DIR / "reddit_city_audit.csv"
REDDIT_CITY_AUDIT_ERRORS_PATH = RAW_DIR / "reddit_city_audit_errors.csv"
REDDIT_CLEAN_PATH = INTERIM_DIR / "reddit_posts_clean.csv"
REDDIT_MONTHLY_PATH = INTERIM_DIR / "reddit_monthly_features.csv"
REDDIT_DUMP_DIR = RAW_DIR / "reddit_dump_submissions"
REDDIT_DUMP_FILE_GLOB = "**/*_submissions.zst"
REDDIT_TARGET_SUBREDDITS = [
    "travel",
    "solotravel",
    "onebag",
    "backpacking",
    "europe",
    "europetravel",
    "askreddit",
    "askeurope",
    "cityporn",
    "iwantout",
    "digitalnomad",
    "paris",
    "rome",
    "barcelona",
    "amsterdam",
    "prague",
    "vienna",
    "budapest",
    "lisbon",
    "berlin",
    "athens",
    "london",
    "madrid",
    "istanbul",
    "copenhagen",
    "stockholm",
    "dublin",
    "brussels",
    "warsaw",
    "florence",
    "venice",
]

MODEL_DATASET_PATH = PROCESSED_DIR / "model_dataset.csv"
PREDICTIONS_PATH = REPORT_DIR / "model_predictions.csv"
METRICS_PATH = REPORT_DIR / "model_metrics.csv"

MIN_REDDIT_SCORE = 1
MIN_TEXT_LENGTH = 20
REQUEST_SLEEP_SECONDS = 1.0
MAX_POSTS_PER_CITY = None
AUDIT_POSTS_PER_CITY = 500

# Optional explicit proxy URL, e.g. "http://username:password@proxy.company.com:8080"
PROXY_URL = None
