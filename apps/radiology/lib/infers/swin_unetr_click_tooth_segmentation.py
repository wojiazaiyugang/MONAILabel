import shutil
from pathlib import Path
from typing import Sequence, Callable, Union, Dict, Tuple, Any, List

import torch
from infer import Image3D, Point3D, click_tooth_segmentation_swin_unetr

from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.infer import InferType

model = click_tooth_segmentation_swin_unetr


class SwinUnetrClickToothSegmentation(InferTask):
    def __int__(self, path, network, type, labels, dimension, description):
        super().__init__(path=path,
                         network=network,
                         type=type,
                         labels=labels,
                         dimension=dimension,
                         description=description)

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return model.get_pre_transforms()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return model.get_post_transforms()

    def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        """
        重写call方法，自行推理
        """
        image_path = Path(request["image"])
        image = Image3D.from_path(Path(request["image"]))
        output_file = Path("/tmp/seg.nii.gz")
        msg = ""
        if self.type == InferType.SEGMENTATION:
            init_label_path = image_path.parent.joinpath("labels").joinpath("final").joinpath(image_path.name)  # 已有的分割结果
            pre_label_path = image_path.parent.joinpath("labels").joinpath("pre").joinpath(image_path.name)  # 预分割结果
            if init_label_path.exists():
                msg = "已有分割结果，直接返回"
                shutil.copy(init_label_path, output_file)
            elif pre_label_path.exists():
                msg = "没有分割结果，返回预分割结果"
                shutil.copy(pre_label_path, output_file)
            else:
                msg = "没有预分割结果，返回空空标签"
                # 生成一个空的标签
                empty_label = Image3D(torch.zeros((10, 10, 10), dtype=torch.int64))
                empty_label.save(output_file)
        elif self.type == InferType.DEEPGROW:
            click: List[int] = request["foreground"][-1]
            center = Point3D(x=click[0], y=click[1], z=click[2]).to_int()
            result = model.infer(image=image, click_point=center)
            seg = Image3D(result[1]).re_spacing(spacing=image.spacing, mode="bilinear").gaussian_smooth(sigma=1).as_discrete(threshold=0.5)
            seg.save(output_file)
        print(f"图片{image_path} => {msg}")
        return str(output_file), {"label_names": self.labels}
