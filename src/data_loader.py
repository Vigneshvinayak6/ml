"""Public data-loading API required by the project layout."""

from red_wine_quality.data import ALTERNATIVE_URL, TARGET, UCI_URL, download_dataset, load_data

__all__ = ["ALTERNATIVE_URL", "TARGET", "UCI_URL", "download_dataset", "load_data"]
