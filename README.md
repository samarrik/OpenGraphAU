## OpenGraphAU (inference-only)

Minimal, PyTorch-like API to instantiate OpenGraphAU models, load weights, run inference, and list output labels.

This project is a streamlined fork of the original OpenGraphAU and ME-GraphAU codebases. Please see and cite the original work:
- Original OpenGraphAU repository: [lingjivoo/OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU)
- ME-GraphAU (IJCAI 2022) repository: [CVI-SZU/ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU)
- Paper: Luo et al., “Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition,” IJCAI-ECAI 2022 ([repo link](https://github.com/CVI-SZU/ME-GraphAU))

### Install
- From source: `pip install -e .`
- Or use `requirements.txt` in your environment.

### Interfaces
- Low-level (raw PyTorch): you get a `torch.nn.Module` and must preprocess tensors yourself (CHW/BCHW normalized with ImageNet mean/std, sized exactly 224×224).
- High-level (`OpenGraphAUPredictor`): pass a PIL image (or a ready 224×224 tensor); preprocessing is applied internally for PIL via torchvision transforms.
- Labels: `list_labels()` for AU codes, `list_label_full_names()` for human names, `label_name_map()` for dict, or `labels_info()` for list of (code, name).

### Low-level usage (handcrafted preprocessing)```python
import torch
import torchvision.transforms as T
from opengraphau import load_model

# Create raw model and move to device. Optionally set local backbone weights dir via backbone_pretrain_dir
model = load_model(stage=2, backbone="resnet50", weights_path="/path/to/model.pth", device="cuda", backbone_pretrain_dir="/path/to/pretrain_models").eval()

# Prepare tensor(s) by hand: BCHW uint8 [0,255], size exactly 224x224
x = torch.randint(0, 256, (8, 3, 224, 224), dtype=torch.uint8)

# Handcrafted preprocessing pipeline (no resize/crop here)
preprocess = T.Compose([
    T.ConvertImageDtype(torch.float32),
    T.Lambda(lambda t: t/255.0),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

x = preprocess(x).to("cuda")

with torch.no_grad():
    logits = model(x)  # shape (B, 41)
```

### High-level usage
```python
from PIL import Image
from opengraphau.inference import OpenGraphAUPredictor

# By default, threshold=0.0 selects AUs with non-negative logits
predictor = OpenGraphAUPredictor(stage=2, backbone="resnet50", weights_path="FILE_ID", device="cpu", threshold=0.0)

# PIL image (resize+center-crop+ToTensor+normalize are applied internally)
image = Image.open("/path/to/image.jpg").convert("RGB")
logits, active_aus = predictor.predict(image)

# Optional: tensor path (CHW or BCHW, must be 224x224); normalization handled internally
logits_batch = predictor.predict_tensor(torch.rand(8, 3, 224, 224))
```

### Preprocessing and normalization
- Inputs should be RGB and normalized with ImageNet mean/std used by torchvision backbones: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- Low-level: tensors must be sized 224×224; Divide, Normalize.
- High-level: `predict(image)` applies resize+center-crop+ToTensor+normalize for PIL; `predict_tensor(x)` expects tensors already sized 224×224 and normalizes them.
- Move the model/tensors to device as needed; `.to(device)` is standard [PyTorch](https://discuss.pytorch.org/t/understanding-model-to-device/123662).

### Labels
```python
from opengraphau import list_labels, list_label_full_names, label_name_map, labels_info

codes = list_labels()                 # ['AU1', 'AU2', ...]
full_names = list_label_full_names()  # ['Inner brow raiser', 'Outer brow raiser', ...]
code_to_name = label_name_map()       # {'AU1': 'Inner brow raiser', ...}
code_name_pairs = labels_info()       # [('AU1','Inner brow raiser'), ...]
```

### Notes
- Backbone pretrained weights directory can be set via:
  - `backbone_pretrain_dir` argument to `load_model(...)`
  - or env var `OPENGRAPHAU_PRETRAIN_DIR`
  Else defaults to `~/.cache/opengraphau/pretrain_models`.
- Outputs are 41 logits in the order of `list_labels()`; no sigmoid is applied.
- Backbones: ResNet-18/50/101 and Swin-Tiny/Small/Base.
- Inference-only repository.



