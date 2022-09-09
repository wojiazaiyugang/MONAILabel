from pathlib import Path
from typing import Sequence, Callable, Union, Dict, Tuple, Any

import torch
from infer import BBox3D, DetectResult3D, Image3D, Point3D
from infer.model_zoo import deep_grow_tooth_segmentater
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.interfaces.tasks.infer import InferTask


class DeepGrowToothSegmentation(InferTask):
    def __init__(
            self,
            path,
            network=None,
            type=InferType.DEEPGROW,
            labels=None,
            dimension=2,
            description="A pre-trained DeepGrow model based on UNET",
            spatial_size=(128, 128, 128),
            model_size=(256, 256),
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
        self.model_size = model_size

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return deep_grow_tooth_segmentater.get_pre_transforms()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return deep_grow_tooth_segmentater.get_post_transforms()

    def __call__(self, request):
        image = Image3D.from_file(Path(request["image"]))
        guidance = []
        for g in request["foreground"]:
            guidance.append(Point3D(x=g[0], y=g[1], z=g[2]))
        result = deep_grow_tooth_segmentater.infer(image=image, fg_guidance=guidance, bg_guidance=[])
        seg = Image3D(result)
        output_file = Path("/tmp/seg2.nii.gz")
        seg.save(output_file)
        return str(output_file), {"label_names": self.labels}


