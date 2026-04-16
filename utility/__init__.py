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

"""
Utility module with lazy loading for Isaac Sim dependencies.
"""

# Always available: Logger (no Isaac dependency)
from .logger import Logger

# Lazy loading state
_isaac_modules_loaded = False
_common_utils_module = None
_camera_utils_module = None
_prim_utils_module = None


def _ensure_isaac_modules():
    """Load Isaac Sim dependent modules on first use."""
    global _isaac_modules_loaded, _common_utils_module, _camera_utils_module, _prim_utils_module

    if not _isaac_modules_loaded:
        try:
            from . import common_utils as _cu
            from . import camera_utils as _cam
            from . import prim_utils as _pu

            _common_utils_module = _cu
            _camera_utils_module = _cam
            _prim_utils_module = _pu
            _isaac_modules_loaded = True
        except ImportError as e:
            raise ImportError(
                "Failed to import Isaac Sim dependent modules. "
                "These utilities require Isaac Sim environment. "
                f"Original error: {e}"
            ) from e


def get_camera_image(*args, **kwargs):
    """
    Lazy-loaded wrapper for get_camera_image from camera_utils.
    Only imports Isaac Sim dependencies when actually called.
    """
    _ensure_isaac_modules()
    return _camera_utils_module.get_camera_image(*args, **kwargs)


# Lazy-loaded class proxies
class _LazyProxy:
    """Base class for lazy-loaded proxies that delegate to common_utils classes."""

    _target_class_name = None

    def __getattribute__(self, name):
        if name.startswith("_") or name in ("__class__", "__dict__"):
            return object.__getattribute__(self, name)
        _ensure_isaac_modules()
        target_class_name = object.__getattribute__(self, "_target_class_name")
        target_class = getattr(_common_utils_module, target_class_name)
        return getattr(target_class, name)


class CommonUtils(_LazyProxy):
    """Lazy-loaded proxy for CommonUtils."""

    _target_class_name = "CommonUtils"


class TransformUtils(_LazyProxy):
    """Lazy-loaded proxy for TransformUtils."""

    _target_class_name = "TransformUtils"


class MathUtils(_LazyProxy):
    """Lazy-loaded proxy for MathUtils."""

    _target_class_name = "MathUtils"


class FilterUtils(_LazyProxy):
    """Lazy-loaded proxy for FilterUtils."""

    _target_class_name = "FilterUtils"


class ValidationUtils(_LazyProxy):
    """Lazy-loaded proxy for ValidationUtils."""

    _target_class_name = "ValidationUtils"


class PrimUtils:
    """Lazy-loaded proxy for PrimUtils (delegates to prim_utils module)."""

    def __getattribute__(self, name):
        if name.startswith("_") or name in ("__class__", "__dict__"):
            return object.__getattribute__(self, name)
        _ensure_isaac_modules()
        target_class = getattr(_prim_utils_module, "PrimUtils")
        return getattr(target_class, name)


# Instantiate singleton proxies
CommonUtils = CommonUtils()
TransformUtils = TransformUtils()
MathUtils = MathUtils()
FilterUtils = FilterUtils()
ValidationUtils = ValidationUtils()
PrimUtils = PrimUtils()


# Lazy-loaded function wrappers for standalone functions
def load_yaml(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.load_yaml(*args, **kwargs)


def load_json(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.load_json(*args, **kwargs)


def save_yaml(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.save_yaml(*args, **kwargs)


def save_json(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.save_json(*args, **kwargs)


def quat_wxyz_to_xyzw(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.quat_wxyz_to_xyzw(*args, **kwargs)


def quat_xyzw_to_wxyz(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.quat_xyzw_to_wxyz(*args, **kwargs)


def euler_to_quat(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.euler_to_quat(*args, **kwargs)


def quat_to_euler(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.quat_to_euler(*args, **kwargs)


def pose_to_matrix(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.pose_to_matrix(*args, **kwargs)


def matrix_to_pose(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.matrix_to_pose(*args, **kwargs)


def normalize(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.normalize(*args, **kwargs)


def lerp(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.lerp(*args, **kwargs)


def slerp(*args, **kwargs):
    _ensure_isaac_modules()
    return _common_utils_module.slerp(*args, **kwargs)


__all__ = [
    "Logger",
    "get_camera_image",
    "CommonUtils",
    "TransformUtils",
    "MathUtils",
    "PrimUtils",
    "FilterUtils",
    "ValidationUtils",
    "load_yaml",
    "load_json",
    "save_yaml",
    "save_json",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "euler_to_quat",
    "quat_to_euler",
    "pose_to_matrix",
    "matrix_to_pose",
    "normalize",
    "lerp",
    "slerp",
]
