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

import logging
import random
import time

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy

logger = logging.getLogger(__name__)


class Order(Strategy):
    """
    Consider implementing a random strategy for active learning
    """

    def __init__(self):
        self.index = -1
        super().__init__("顺序读取数据")

    def __call__(self, request, datastore: Datastore):
        strategy = request.get("strategy")
        offset = 1 if strategy == "next" else -1
        images = list(sorted(datastore.get_unlabeled_images() + datastore.get_labeled_images()))
        index = self.index + offset
        if index >= len(images):
            index = 0
        if index < 0:
            index = len(images) - 1
        self.index = index
        return images[index]
