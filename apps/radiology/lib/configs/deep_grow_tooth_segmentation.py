from pathlib import Path
from typing import Dict, Any, Union, Optional

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask, InferType
from apps.radiology.lib.infers.deep_grow_toothsegmentation import \
    DeepGrowToothSegmentation as DeepGrowToothSegmentationInferTask

from infer.model_zoo import deep_grow_tooth_segmentater

from monailabel.interfaces.tasks.train import TrainTask
from monai.networks.nets import BasicUNet


class DeepGrowToothSegmentation(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)
        self.labels = [
            "tooth"
        ]
        self.path = [
            str(deep_grow_tooth_segmentater.model_config.model_file),
            str(Path(self.model_dir).joinpath("deep_grow_tooth_jawbone_segmentater_publish.pth"))
        ]

        # Network
        self.network = BasicUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            features=(32, 64, 128, 256, 512, 32),
        )

    def trainer(self) -> Optional[TrainTask]:
        return None

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return DeepGrowToothSegmentationInferTask(path=self.path,
                                                  network=self.network,
                                                  labels=self.labels)
