from pathlib import Path
from typing import Sequence, Callable, Union, Dict, Tuple, Any, List

from infer import Image3D, Point3D, click_tooth_segmentation_swin_unetr

from monailabel.interfaces.tasks.infer import InferTask

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
        image = Image3D.from_path(Path(request["image"]))
        click: List[int] = request["foreground"][-1]
        center = Point3D(x=click[0], y=click[1], z=click[2]).to_int()
        result = model.infer(image=image, click_point=center)
        seg = Image3D(result[1])
        seg.re_spacing(spacing=image.spacing, mode="bilinear")
        seg.gaussian_smooth(sigma=1)
        seg.as_discrete(threshold=0.5)
        output_file = Path("/tmp/seg.nii.gz")
        seg.save(output_file)
        return str(output_file), {"label_names": self.labels}
