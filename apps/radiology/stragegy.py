# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from datetime import datetime
import logging
from pathlib import Path

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy

logger = logging.getLogger(__name__)


class Order(Strategy):
    """
    Consider implementing a random strategy for active learning
    """

    def __init__(self):
        self.index = -1
        self.extra_info = ""  # 额外信息，用于显示在前端
        super().__init__("顺序读取数据")

    @staticmethod
    def pad_string(s: str) -> str:
        return s + " " * (150 - len(s))

    def __call__(self, request, datastore: Datastore):
        strategy = request.get("strategy")
        labeled_image_ids = list(datastore.get_labeled_images())
        unlabeled_image_ids = list(datastore.get_unlabeled_images())
        image_ids = list(sorted(labeled_image_ids + unlabeled_image_ids))
        if strategy == "Next":
            if self.index + 1 >= len(image_ids):
                return None
            self.index += 1
        elif strategy == "Previous":
            if self.index - 1 < 0:
                return None
            self.index -= 1
        elif strategy == "First":
            self.index = 0
        elif strategy == "Last":
            self.index = len(image_ids) - 1
        select_image_id = image_ids[self.index]
        select_image_file = Path(datastore.get_image_info(image_ids[self.index])["path"])
        label_file = select_image_file.parent.joinpath("labels").joinpath("final").joinpath(select_image_file.name)
        self.extra_info = (
            self.pad_string("") +
            self.pad_string(f"""待标注数据共计{len(image_ids)}份，已标注{len(labeled_image_ids)}份，未标注{len(unlabeled_image_ids)}份""") +
            self.pad_string(f"""当前选择第{self.index + 1}份样本{select_image_file.name} """) +
            self.pad_string(f"""当前样本标注状态：{"已标注" if label_file.exists() else "未标注"} {"标注时间:" + str(datetime.fromtimestamp(os.path.getmtime(label_file))) if label_file.exists() else ""} """)
        )
        self.extra_info += self.pad_string(f"所有数据标注状态:")
        for index, image_id in enumerate(image_ids):
            image_file = Path(datastore.get_image_info(image_id)["path"])
            label_file = image_file.parent.joinpath("labels").joinpath("final").joinpath(image_file.name)
            self.extra_info += self.pad_string(f"""{"=>" if index == self.index else ". "} 第{index+1}份样本({image_file.name:75})：{"已标注" if label_file.exists() else "未标注"} {"标注时间:" + str(datetime.fromtimestamp(os.path.getmtime(label_file))) if label_file.exists() else ""} """)
        return select_image_id
