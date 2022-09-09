from pathlib import Path
from typing import Sequence, Callable

import torch
from infer import deep_edit_tooth_segmentater, Image3D, Point3D

from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.infer import InferType


class DeepEditToothInstanceSegmentation(InferTask):
    def __init__(
            self,
            path,
            network=None,
            type=InferType.DEEPEDIT,
            labels=None,
            dimension=3,
            spatial_size=(160, 160, 160),
            target_spacing=(0.25, 0.25, 0.25),
            description="A DeepEdit model for volumetric (3D) segmentation over 3D Images",
            **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            **kwargs,
        )
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return deep_edit_tooth_segmentater.get_pre_transforms()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return deep_edit_tooth_segmentater.get_post_transforms()

    def __call__(self, request):
        image = Image3D.from_path(Path(request["image"]))
        guidance = {}
        for label_key in self.labels:
            label_guidance = request.get(label_key, [])
            for index, point in enumerate(label_guidance):
                label_guidance[index] = Point3D(x=point[0], y=point[1], z=point[2])
            guidance[label_key] = label_guidance
        result = deep_edit_tooth_segmentater.infer(image=image, guidance=guidance)
        seg = Image3D(torch.argmax(result, dim=0))
        output_file = Path("/tmp/seg.nii.gz")
        seg.save(output_file)
        return str(output_file), {"label_names": self.labels}


