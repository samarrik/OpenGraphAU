from .utils import list_labels
from .inference import OpenGraphAUPredictor

__all__ = [
    "load_model",
    "list_backbones",
    "list_stages",
    "list_labels",
    "list_label_full_names",
    "label_name_map",
    "labels_info",
    "OpenGraphAUPredictor",
]

__version__ = "0.1.0"


def load_model(*args, **kwargs):
    from .factory import load_model as _impl
    return _impl(*args, **kwargs)


def list_backbones():
    from .factory import list_backbones as _impl
    return _impl()


def list_stages():
    from .factory import list_stages as _impl
    return _impl()


# Re-export label helpers lazily to avoid import cycles

def list_label_full_names():
    from .utils import list_label_full_names as _impl
    return _impl()


def label_name_map():
    from .utils import label_name_map as _impl
    return _impl()


def labels_info():
    from .utils import labels_info as _impl
    return _impl() 