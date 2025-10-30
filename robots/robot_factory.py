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
from typing import Dict, Type
from .base_robot import BaseRobot
from .a2_robot import A2Robot
from comm_config.robot_config import RobotConfig
from utility.logger import Logger


class RobotFactory:

    _robot_classes: Dict[str, Type[BaseRobot]] = {
        "A2": A2Robot,
    }

    @classmethod
    def create_robot(cls, robot_type: str, config: RobotConfig) -> BaseRobot:
        if robot_type not in cls._robot_classes:
            available_types = ", ".join(cls._robot_classes.keys())
            raise ValueError(
                f"Unknown robot type: {robot_type}. Available types: {available_types}"
            )
        robot_class = cls._robot_classes[robot_type]
        robot = robot_class(config)

        Logger.info(f"Created robot of type: {robot_type}")
        return robot
