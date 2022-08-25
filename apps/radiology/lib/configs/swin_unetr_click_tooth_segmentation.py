from pathlib import Path
from typing import Dict, Any, Union, Optional

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask, InferType
from apps.radiology.lib.infers.swin_unetr_click_tooth_segmentation import \
    SwinUnetrClickToothSegmentation as SwinUnetrClickToothSegmentationInferTask

from infer import swin_unetr_click_tooth_segmentater

from monailabel.interfaces.tasks.train import TrainTask


class SwinUnetrClickToothSegmentation(TaskConfig):
    def __init__(self):
        super().__init__()

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.labels = {
            "background": 0,
            "tooth": 1
        }

        self.path = [
            str(swin_unetr_click_tooth_segmentater.model_config.model_file),
            str(Path(self.model_dir).joinpath("swin_unetr_click_tooth_segmentater_publish.pth"))
        ]

        self.network = swin_unetr_click_tooth_segmentater.model

    def trainer(self) -> Optional[TrainTask]:
        return None

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return SwinUnetrClickToothSegmentationInferTask(path=self.path,
                                                        network=self.network,
                                                        type=InferType.DEEPEDIT,
                                                        labels=self.labels,
                                                        dimension=3,
                                                        description="A SwinUnetrClickToothSegmentation")
