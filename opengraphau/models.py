from typing import Optional

from .model import ANFL as anfl_mod
from .model import MEFL as mefl_mod


def build_model(
    stage: int,
    backbone: str,
    num_main_classes: int,
    num_sub_classes: int,
    neighbor_num: int = 4,
    metric: str = "dots",
    pretrain_dir: Optional[str] = None,
):
    if stage == 1:
        return anfl_mod.MEFARG(
            num_main_classes=num_main_classes,
            num_sub_classes=num_sub_classes,
            backbone=backbone,
            neighbor_num=neighbor_num,
            metric=metric,
            pretrain_dir=pretrain_dir,
        )
    elif stage == 2:
        return mefl_mod.MEFARG(
            num_main_classes=num_main_classes,
            num_sub_classes=num_sub_classes,
            backbone=backbone,
            pretrain_dir=pretrain_dir,
        )
    else:
        raise ValueError(f"Unsupported stage: {stage}. Use 1 or 2.") 