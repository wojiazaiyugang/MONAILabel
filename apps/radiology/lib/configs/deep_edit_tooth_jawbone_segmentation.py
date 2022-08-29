from pathlib import Path
from typing import Dict, Any, Union, Optional

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask, InferType
from apps.radiology.lib.infers.deep_edit_tooth_jawbone_segmentation import \
    DeepEditToothJawboneSegmentation as DeepEditToothJawboneSegmentationInferTask

from infer import deep_edit_tooth_jawbone_segmentater

from monailabel.interfaces.tasks.train import TrainTask


class DeepEditToothJawboneSegmentation(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)
        self.labels = deep_edit_tooth_jawbone_segmentater.model_config.labels
        self.path = [
            str(deep_edit_tooth_jawbone_segmentater.model_config.model_file),
            str(Path(self.model_dir).joinpath("deep_edit_tooth_jawbone_segmentater_publish.pth"))
        ]
        self.target_spacing = deep_edit_tooth_jawbone_segmentater.model_config.spacing
        self.spatial_size = deep_edit_tooth_jawbone_segmentater.model_config.image_size
        self.network = deep_edit_tooth_jawbone_segmentater.model

    def trainer(self) -> Optional[TrainTask]:
        return None

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return {
            self.name: DeepEditToothJawboneSegmentationInferTask(path=self.path,
                                                                 network=self.network,
                                                                 labels=self.labels,
                                                                 spatial_size=self.spatial_size,
                                                                 target_spacing=self.target_spacing),
            f"{self.name}_auto_seg": DeepEditToothJawboneSegmentationInferTask(path=self.path,
                                                                               network=self.network,
                                                                               labels=self.labels,
                                                                               spatial_size=self.spatial_size,
                                                                               target_spacing=self.target_spacing,
                                                                               type=InferType.SEGMENTATION)

        }
