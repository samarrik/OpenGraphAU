from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image

from .factory import load_model
from .utils import image_eval


class OpenGraphAUPredictor:
    def __init__(
        self,
        stage: int = 1,
        backbone: str = "resnet50",
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.0,
        crop_size: int = 224,
        model: Optional[torch.nn.Module] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        if model is not None:
            self.model = model
            if device is not None:
                self.model.to(torch.device(device))
        else:
            self.model = load_model(stage=stage, backbone=backbone, weights_path=weights_path, device=device, model_dir=model_dir)
        self.model.eval()
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = image_eval(crop_size=crop_size)
        self.threshold = threshold
        # Labels corresponding to output feature indices
        from .utils import list_labels as _list_labels
        self.label_names: List[str] = _list_labels()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[np.ndarray, List[str]]:
        """
        Run inference on a PIL image.
        Returns:
            logits: numpy array of shape (41,)
            active_aus: list of AU label names where logit >= threshold
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        logits_np = logits.squeeze(0).cpu().numpy()
        active_idx = np.where(logits_np >= self.threshold)[0]
        active_aus = [self.label_names[i] for i in active_idx]
        return logits_np, active_aus

    @torch.no_grad()
    def predict_file(self, image_path: str) -> Tuple[np.ndarray, List[str]]:
        image = Image.open(image_path).convert("RGB")
        return self.predict(image)

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor) -> np.ndarray:
        """
        Accept CHW (3,224,224) or BCHW (B,3,224,224) tensor in [0,1] or [0,255].
        No resizing/cropping is performed here; only ImageNet normalization.
        Returns logits as numpy array: shape (41,) for single image or (B,41) for batch.
        """
        from .utils import tensor_preprocess
        x = tensor_preprocess(x).to(self.device)
        logits = self.model(x)
        return logits.cpu().numpy()

    def labels(self) -> List[str]:
        return self.label_names 