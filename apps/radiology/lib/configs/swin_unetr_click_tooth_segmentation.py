from pathlib import Path
from typing import Dict, Any, Union, Optional

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask, InferType
from apps.radiology.lib.infers.swin_unetr_click_tooth_segmentation import \
    SwinUnetrClickToothSegmentation as SwinUnetrClickToothSegmentationInferTask, model

from monailabel.interfaces.tasks.train import TrainTask


class SwinUnetrClickToothSegmentation(TaskConfig):
    def __init__(self):
        super().__init__()

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.labels = {
            "background": 0,
        }

        for area in range(1, 5):  # 牙齿象限1-8
            for tooth_index in range(1, 9):
                # if area >= 5 and tooth_index >= 6:
                #     continue
                self.labels[f"tooth_{area}{tooth_index}"] = area * 10 + tooth_index
        # self.labels["tooth_unknown"] = 100

        self.path = [
            str(model.model_config.model_file),
            str(Path(self.model_dir).joinpath("swin_unetr_click_tooth_segmentater_publish.pth"))
        ]

        self.network = model.model

    def trainer(self) -> Optional[TrainTask]:
        return None

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return {
            self.name: SwinUnetrClickToothSegmentationInferTask(path=self.path,
                                                                network=self.network,
                                                                type=InferType.DEEPGROW,
                                                                labels=self.labels,
                                                                dimension=3,
                                                                description="A SwinUnetrClickToothSegmentation"),
            "init": SwinUnetrClickToothSegmentationInferTask(path=self.path,
                                                             network=self.network,
                                                             type=InferType.SEGMENTATION,
                                                             labels=self.labels,
                                                             dimension=3,
                                                             description="A SwinUnetrClickToothSegmentation"),
        }
