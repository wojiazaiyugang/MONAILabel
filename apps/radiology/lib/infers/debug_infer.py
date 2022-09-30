import shutil
from pathlib import Path
from typing import Sequence, Callable, Union, Dict, Tuple, Any, List

import torch
from infer import Image3D, Point3D, click_tooth_segmentation_swin_unetr

from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.infer import InferType


class Debug(InferTask):
    def __init__(self, path, network, type, labels, dimension, description):
        super().__init__(path=path,
                         network=network,
                         type=type,
                         labels=labels,
                         dimension=dimension,
                         description=description)
        self.image, self.image_path = None, None

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return model.get_pre_transforms()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return model.get_post_transforms()

    def __call__(self, request) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        """
        重写call方法，自行推理
        """
        image_path = Path(request["image"])
        output_file = Path("/tmp/seg.nii.gz")
        if image_path == self.image_path:
            image = self.image.clone()
            self.image_path = image_path
            self.image = image
        else:
            image = Image3D.from_path(image_path)
        result = click_tooth_segmentation_swin_unetr.infer(image=image, click_point=Point3D(x=136, y=343, z=259))
        label = torch.argmax(result, dim=0)
        Image3D(label).save(output_file)
        return str(output_file), {"label_names": self.labels}
