"""Download and load the UCI Red Wine Quality dataset."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ wine-quality/winequality-red.csv".replace(" ", "")
ALTERNATIVE_URL = (
    "https://raw.githubusercontent.com/selva86/datasets/master/winequality-red.csv"
)
TARGET = "quality"


def download_dataset(
    destination: str | Path = "data/winequality-red.csv",
    urls: tuple[str, ...] = (UCI_URL, ALTERNATIVE_URL),
    timeout: int = 30,
) -> Path:
    """Download the semicolon-delimited UCI data, trying a maintained mirror.

    The original dataset is the UCI Machine Learning Repository's
    ``wine-quality/winequality-red.csv`` (Cortez et al., 2009).  If UCI is
    unavailable, a public GitHub mirror is attempted. Existing files are
    never overwritten.
    """
    path = Path(destination)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            request = Request(url, headers={"User-Agent": "red-wine-quality/0.1"})
            with urlopen(request, timeout=timeout) as response:
                content = response.read()
            if b"fixed acidity" not in content[:5000]:
                raise ValueError("response does not look like wine-quality CSV")
            path.write_bytes(content)
            return path
        except Exception as exc:  # pragma: no cover - depends on network
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download dataset. " + " | ".join(errors))


def load_data(path: str | Path = "data/winequality-red.csv", download: bool = True) -> pd.DataFrame:
    """Load data and validate the expected schema."""
    path = Path(path)
    if download and not path.exists():
        download_dataset(path)
    frame = pd.read_csv(path, sep=";")
    if TARGET not in frame or frame.shape[1] < 3:
        raise ValueError("Unexpected dataset schema; expected quality target.")
    return frame
