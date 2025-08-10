import os
from typing import Optional
import torch

from .models import build_model


_DEFAULT_NUM_MAIN = 27
_DEFAULT_NUM_SUB = 14


def list_stages():
    return [1, 2]


def list_backbones():
    return [
        "resnet18",
        "resnet50",
        "resnet101",
        "swin_transformer_tiny",
        "swin_transformer_small",
        "swin_transformer_base",
    ]


def _is_gdrive_ref(ref: str) -> bool:
    ref = (ref or "").strip()
    if not ref:
        return False
    return (
        "drive.google.com" in ref
        or ref.startswith("gdrive:")
        or (len(ref) >= 10 and "/" not in ref and "." not in ref)  # heuristic for bare file id
    )


def _extract_gdrive_id(ref: str) -> Optional[str]:
    ref = (ref or "").strip()
    if not ref:
        return None
    if ref.startswith("gdrive:"):
        return ref.split(":", 1)[1]
    if "drive.google.com" in ref:
        # Common patterns:
        # - https://drive.google.com/file/d/<ID>/view?usp=sharing
        # - https://drive.google.com/uc?id=<ID>&export=download
        # - https://drive.google.com/open?id=<ID>
        import re
        m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", ref)
        if m:
            return m.group(1)
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", ref)
        if m:
            return m.group(1)
    if "/" not in ref and "." not in ref and len(ref) >= 10:
        return ref
    return None


def _looks_like_html_or_too_small(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(4096)
        lhead = head.lower()
        if head[:1] == b"<" or b"<html" in lhead or b"<!doctype" in lhead or b"google drive" in lhead or b"quota" in lhead or b"signin" in lhead:
            return True
        if size < 5 * 1024 * 1024:
            return True
    except Exception:
        return True
    return False


def _download_gdrive(ref: str, cache_dir: Optional[str] = None, filename: Optional[str] = None) -> str:
    try:
        import gdown  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "gdown is required to download weights from Google Drive. Install with `pip install gdown`."
        ) from exc

    cache_root = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "opengraphau")
    os.makedirs(cache_root, exist_ok=True)

    output = os.path.join(cache_root, filename) if filename else None

    # If a valid cached file already exists, return it without re-downloading
    if output and os.path.exists(output) and not _looks_like_html_or_too_small(output):
        return output

    # Remove invalid cached file before downloading
    if output and os.path.exists(output) and _looks_like_html_or_too_small(output):
        try:
            os.remove(output)
        except Exception:
            pass

    file_id = _extract_gdrive_id(ref)
    if file_id is not None:
        out_path = gdown.download(id=file_id, output=output, quiet=False, use_cookies=False)
    else:
        out_path = gdown.download(url=ref, output=output, quiet=False, fuzzy=True, use_cookies=False)

    if not out_path or not os.path.exists(out_path) or _looks_like_html_or_too_small(out_path):
        if file_id is None:
            file_id = _extract_gdrive_id(ref)
        if file_id is not None:
            if output and os.path.exists(output):
                try:
                    os.remove(output)
                except Exception:
                    pass
            alt_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            out_path = gdown.download(url=alt_url, output=output, quiet=False, use_cookies=False)

    if not out_path or not os.path.exists(out_path) or _looks_like_html_or_too_small(out_path):
        raise RuntimeError(
            "Failed to download a valid checkpoint from Google Drive. "
            "Ensure the link is publicly accessible and not quota-limited, or pass the bare file id."
        )

    return out_path


def _torch_load_compat(path: str, map_location: Optional[str] = "cpu"):
    """Load torch checkpoint across versions (handles PyTorch 2.6 weights_only change).
    Provide clearer diagnostics if the file is an HTML/permission page.
    """
    try:
        return torch.load(path, map_location=torch.device(map_location), weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location=torch.device(map_location))
    except Exception as exc:
        try:
            if _looks_like_html_or_too_small(path):
                raise RuntimeError(
                    "Downloaded file is not a valid PyTorch checkpoint (looks like an HTML/Drive page). "
                    "Ensure the link is public, avoid quota limits, or pass a bare file id."
                ) from exc
        except Exception:
            pass
        raise


def _extract_state_dict(obj: object):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj


def load_model(
    stage: int = 1,
    backbone: str = "resnet50",
    num_main_classes: int = _DEFAULT_NUM_MAIN,
    num_sub_classes: int = _DEFAULT_NUM_SUB,
    neighbor_num: int = 4,
    metric: str = "dots",
    weights_path: Optional[str] = None,
    map_location: Optional[str] = "cpu",
    weights_cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    backbone_pretrain_dir: Optional[str] = None,
) -> torch.nn.Module:
    """Create a MEFARG model for the requested stage/backbone and optionally load weights.

    Args:
        stage: 1 uses `model.ANFL.MEFARG`, 2 uses `model.MEFL.MEFARG`.
        backbone: One of list_backbones().
        num_main_classes: Defaults to 27.
        num_sub_classes: Defaults to 14.
        neighbor_num: Only used by stage 1.
        metric: Only used by stage 1.
        weights_path: Local path, Google Drive URL, or file id to load. If Google Drive, uses gdown.
        map_location: torch.load map_location for checkpoint. If `device` is given, this is ignored.
        weights_cache_dir: Optional directory to cache downloaded weights (defaults to ~/.cache/opengraphau).
        device: Optional torch device string (e.g., 'cuda', 'cuda:0', or 'cpu'). If provided, the model is moved to this device.
        backbone_pretrain_dir: Optional directory containing backbone pretrained weights (e.g., resnet50-19c8e357.pth, swin_*.pth).
            If not provided, falls back to env OPENGRAPHAU_PRETRAIN_DIR or ~/.cache/opengraphau/pretrain_models.
    Returns:
        torch.nn.Module ready for inference. Caller should call .eval() as needed.
    """
    model = build_model(
        stage=stage,
        backbone=backbone,
        num_main_classes=num_main_classes,
        num_sub_classes=num_sub_classes,
        neighbor_num=neighbor_num,
        metric=metric,
        pretrain_dir=backbone_pretrain_dir,
    )

    try:
        from .utils import list_labels as _list_labels
        model.output_labels = _list_labels()
    except Exception:
        pass

    effective_map_location = device if device is not None else map_location

    if weights_path:
        resolved_path = weights_path
        if _is_gdrive_ref(weights_path):
            default_name = f"opengraphau_{backbone}_stage{stage}.pth"
            resolved_path = _download_gdrive(weights_path, cache_dir=weights_cache_dir, filename=default_name)

        ckpt = _torch_load_compat(resolved_path, map_location=effective_map_location)
        state_dict = _extract_state_dict(ckpt)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if isinstance(k, str) and k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    if device is not None:
        model.to(torch.device(device))

    return model 