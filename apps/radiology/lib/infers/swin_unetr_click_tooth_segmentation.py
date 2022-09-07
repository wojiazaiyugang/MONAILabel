import random
from pathlib import Path
from typing import Sequence, Callable, Union, Dict, Tuple, Any, List

import torch
from infer import swin_unetr_click_tooth_segmentater, BBox3D, DetectResult3D, Image3D, Point3D, \
    swin_unetr_click_tooth_segmentater_025

from monailabel.interfaces.tasks.infer import InferTask

model = swin_unetr_click_tooth_segmentater_025


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
        image = Image3D.from_path(Path(request["image"]))
        click: List[int] = request["foreground"][-1]
        image_spacing = image.spacing
        model_spacing = model.model_config.spacing
        ratio = [image_spacing[0] / model_spacing[0], image_spacing[1] / model_spacing[1], image_spacing[2] / model_spacing[2]]
        center = Point3D(x=click[0] * ratio[0], y=click[1] * ratio[1], z=click[2] * ratio[2]).to_int()
        point1 = center.offset(-model.model_config.image_size[0] // 2,
                               -model.model_config.image_size[1] // 2,
                               -model.model_config.image_size[2] // 2)
        point7 = center.offset(model.model_config.image_size[0] // 2,
                               model.model_config.image_size[1] // 2,
                               model.model_config.image_size[2] // 2)
        result = model.infer(image=image,
                             detect_result=DetectResult3D(bbox=BBox3D(point1=point1,
                                                                      point7=point7)))
        seg = Image3D(torch.argmax(result, dim=0))
        output_file = Path("/tmp/seg.nii.gz")
        seg.save(output_file)
        return str(output_file), {"label_names": self.labels}
