from typing import Tuple, List, Union
import numpy as np
from torchvision import transforms
from PIL import Image
import torch

# Canonical label codes and full names in model output order
LABEL_CODES: List[str] = [
    'AU1','AU2','AU4','AU5','AU6','AU7','AU9','AU10','AU11','AU12','AU13','AU14','AU15','AU16','AU17','AU18','AU19','AU20',
    'AU22','AU23','AU24','AU25','AU26','AU27','AU32','AU38','AU39','AUL1','AUR1','AUL2','AUR2','AUL4','AUR4','AUL6','AUR6',
    'AUL10','AUR10','AUL12','AUR12','AUL14','AUR14'
]

LABEL_FULL_NAMES: List[str] = [
    'Inner brow raiser','Outer brow raiser','Brow lowerer','Upper lid raiser','Cheek raiser','Lid tightener',
    'Nose wrinkler','Upper lip raiser','Nasolabial deepener','Lip corner puller','Sharp lip puller','Dimpler',
    'Lip corner depressor','Lower lip depressor','Chin raiser','Lip pucker','Tongue show','Lip stretcher',
    'Lip funneler','Lip tightener','Lip pressor','Lips part','Jaw drop','Mouth stretch','Lip bite',
    'Nostril dilator','Nostril compressor','Left Inner brow raiser','Right Inner brow raiser',
    'Left Outer brow raiser','Right Outer brow raiser','Left Brow lowerer','Right Brow lowerer',
    'Left Cheek raiser','Right Cheek raiser','Left Upper lip raiser','Right Upper lip raiser',
    'Left Nasolabial deepener','Right Nasolabial deepener','Left Dimpler','Right Dimpler'
]


class image_eval(object):
    def __init__(self, img_size: int = 256, crop_size: int = 224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img: Image.Image):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize,
        ])
        return transform(img)


def list_labels() -> List[str]:
    return list(LABEL_CODES)


def list_label_full_names() -> List[str]:
    return list(LABEL_FULL_NAMES)


def label_name_map() -> dict:
    return dict(zip(LABEL_CODES, LABEL_FULL_NAMES))


def labels_info() -> List[tuple[str, str]]:
    return list(zip(LABEL_CODES, LABEL_FULL_NAMES))


def hybrid_prediction_infolist(pred: np.ndarray, thresh: float):
    infostr_pred_probs = {
        'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU5: {:.2f} AU6: {:.2f} AU7: {:.2f} AU9: {:.2f} AU10: {:.2f} AU11: {:.2f} '
        'AU12: {:.2f} AU13: {:.2f} AU14: {:.2f} AU15: {:.2f} AU16: {:.2f} AU17: {:.2f} AU18: {:.2f} AU19: {:.2f} AU20: {:.2f} '
        'AU22: {:.2f} AU23: {:.2f} AU24: {:.2f} AU25: {:.2f} AU26: {:.2f} AU27: {:.2f} AU32: {:.2f} AU38: {:.2f} AU39: {:.2f}'
        ' AUL1: {:.2f} AUR1: {:.2f} AUL2: {:.2f} AUR2: {:.2f} AUL4: {:.2f} AUR4: {:.2f} AUL6: {:.2f} AUR6: {:.2f} AUL10: {:.2f} '
        'AUR10: {:.2f} AUL12: {:.2f} AUR12: {:.2f} AUL14: {:.2f} AUR14: {:.2f}'.format(*[100.*x for x in pred])
    }

    AU_name_lists = LABEL_FULL_NAMES
    AU_indexs = np.where(pred>=thresh)[0]
    AU_prediction = [AU_name_lists[i] for i in AU_indexs]
    infostr_au_pred = {*AU_prediction}
    return infostr_pred_probs, infostr_au_pred



def tensor_preprocess(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Torchvision-style preprocessing using transforms.Compose for tensor inputs.
    - Accepts CHW (3,224,224) or BCHW (B,3,224,224)
    - Supports uint8 [0,255] or float in [0,1] / [0,255]
    - Does NOT resize/crop; expects exactly 224x224 spatial size
    - Applies ConvertImageDtype(float32) and Normalize(mean/std)
    """
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    assert x.dim() == 4 and x.size(1) == 3, "Expected CHW or BCHW with 3 channels"
    b, c, h, w = x.shape
    if h != 224 or w != 224:
        raise ValueError(f"Expected input spatial size 224x224, got {h}x{w}. Resize/crop before calling.")

    # Compose: convert dtype to float32 (scales uint8->float in [0,1]),
    # then scale float inputs in [0,255] if needed, then normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    pipeline = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Lambda(lambda t: t/255.0 if (t.dtype.is_floating_point and torch.isfinite(t).all() and t.max() > 1.5) else t),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Keep device; torchvision transforms operate in-place device-wise
    device = x.device
    x = pipeline(x)
    return x.to(device) 