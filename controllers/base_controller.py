# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseController(ABC):

    @abstractmethod
    def read_data(self) -> Optional[Dict[str, Any]]:

        pass

    @abstractmethod
    def get_buttons(self, controller_data: Dict) -> Dict[str, bool]:

        pass

    @abstractmethod
    def get_axes(self, controller_data: Dict) -> list:

        pass

    @abstractmethod
    def get_position(self, controller_data: Dict) -> Optional[list]:

        pass

    @abstractmethod
    def get_orientation(self, controller_data: Dict) -> Optional[list]:

        pass
