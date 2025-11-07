from enum import Enum


class DownloadStatus(str, Enum):
    INIT = "init"
    DOWNLOADING = "downloading"
    FAILED = "failed"
    READY = "ready"

