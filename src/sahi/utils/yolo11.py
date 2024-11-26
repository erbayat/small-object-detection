import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolo11TestConstants:
    YOLO11N_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    YOLO11N_MODEL_PATH = "tests/data/models/yolo11/yolo11n.pt"

    YOLO11S_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
    YOLO11S_MODEL_PATH = "tests/data/models/yolo11/yolo11s.pt"

    YOLO11M_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
    YOLO11M_MODEL_PATH = "tests/data/models/yolo11/yolo11m.pt"

    YOLO11L_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
    YOLO11L_MODEL_PATH = "tests/data/models/yolo11/yolo11l.pt"

    YOLO11X_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
    YOLO11X_MODEL_PATH = "tests/data/models/yolo11/yolo11x.pt"

    YOLO11N_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
    YOLO11N_SEG_MODEL_PATH = "tests/data/models/yolo11/yolo11n-seg.pt"

    YOLO11S_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt"
    YOLO11S_SEG_MODEL_PATH = "tests/data/models/yolo11/yolo11s-seg.pt"

    YOLO11M_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt"
    YOLO11M_SEG_MODEL_PATH = "tests/data/models/yolo11/yolo11m-seg.pt"

    YOLO11L_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt"
    YOLO11L_SEG_MODEL_PATH = "tests/data/models/yolo11/yolo11l-seg.pt"

    YOLO11X_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt"
    YOLO11X_SEG_MODEL_PATH = "tests/data/models/yolo11/yolo11x-seg.pt"


def download_yolo11n_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11N_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11N_MODEL_URL,
            destination_path,
        )


def download_yolo11s_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11S_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11S_MODEL_URL,
            destination_path,
        )


def download_yolo11m_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11M_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11M_MODEL_URL,
            destination_path,
        )


def download_yolo11l_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11L_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11L_MODEL_URL,
            destination_path,
        )


def download_yolo11x_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11X_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11X_MODEL_URL,
            destination_path,
        )


def download_yolo11n_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11N_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11N_SEG_MODEL_URL,
            destination_path,
        )


def download_yolo11s_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11S_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11S_SEG_MODEL_URL,
            destination_path,
        )


def download_yolo11m_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11M_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11M_SEG_MODEL_URL,
            destination_path,
        )


def download_yolo11l_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11L_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11L_SEG_MODEL_URL,
            destination_path,
        )


def download_yolo11x_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolo11TestConstants.YOLO11X_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolo11TestConstants.YOLO11X_SEG_MODEL_URL,
            destination_path,
        )
