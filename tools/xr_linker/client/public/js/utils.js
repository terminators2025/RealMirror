/**
 * Copyright 2025 ZTE Corporation.
 * All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License"); you may
 *    not use this file except in compliance with the License. You may obtain
 *    a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *    License for the specific language governing permissions and limitations
 *    under the License.
 */

/**
 * Client-side utility functions
 */
import { PRECISION } from './constants.js';

/**
 * Coordinate transformation utility class
 * Converts the WebXR coordinate system to the Isaac Sim coordinate system
 */
export class CoordinateTransformer {
  /**
   * Convert position coordinates
   * @param {Object} position - WebXR position {x, y, z}
   * @returns {Array} Isaac Sim position [x, y, z]
   */
  static transformPosition(position) {
    return [
      parseFloat((-position.z).toFixed(PRECISION.POSITION)),
      parseFloat((-position.x).toFixed(PRECISION.POSITION)),
      parseFloat(position.y.toFixed(PRECISION.POSITION))
    ];
  }

  /**
   * Convert direction quaternion
   * @param {Object} orientation - WebXR direction {x, y, z, w}
   * @returns {Array} Isaac Sim direction [x, y, z, w]
   */
  static transformOrientation(orientation) {
    return [
      parseFloat(orientation.x.toFixed(PRECISION.ORIENTATION)),
      parseFloat(orientation.z.toFixed(PRECISION.ORIENTATION)),
      parseFloat(orientation.y.toFixed(PRECISION.ORIENTATION)),
      parseFloat(orientation.w.toFixed(PRECISION.ORIENTATION))
    ];
  }

  /**
   * Convert pose data
   * @param {Object} transform - WebXR transform object
   * @returns {Object} Converted pose {position, orientation}
   */
  static transformPose(transform) {
    return {
      position: this.transformPosition(transform.position),
      orientation: this.transformOrientation(transform.orientation)
    };
  }
}

/**
 * Handle data processing utility classes
 */
export class ControllerDataProcessor {
  /**
   * Process button data
   * @param {Array} buttons - GamepadButton array
   * @returns {Array} Processed button data
   */
  static processButtons(buttons) {
    return buttons.map(button => ({
      pressed: button.pressed,
      touched: button.touched,
      value: parseFloat(button.value.toFixed(PRECISION.BUTTON_VALUE))
    }));
  }

  /**
   * Process joystick axis data
   * @param {Array} axes - axis value array
   * @returns {Array} Processed axis data
   */
  static processAxes(axes) {
    return axes.map(axis => parseFloat(axis.toFixed(PRECISION.AXES)));
  }

  /**
   * Get controller ID
   * @param {XRInputSource} source - XR input source
   * @returns {string} Controller ID
   */
  static getControllerId(source) {
    if (source.handedness !== 'none') {
      return source.handedness;
    }
    return `controller_${source.profiles.join('_')}`;
  }

  /**
   * Process individual controller data
   * @param {XRInputSource} source - XR input source
   * @param {XRFrame} frame - XR frame
   * @param {XRReferenceSpace} referenceSpace - reference space
   * @returns {Object|null} Processed controller data
   */
  static processController(source, frame, referenceSpace) {
    if (!source.gamepad) {
      return null;
    }

    const controllerData = {
      id: this.getControllerId(source),
      handedness: source.handedness,
      profiles: source.profiles,
      buttons: this.processButtons(source.gamepad.buttons),
      axes: this.processAxes(source.gamepad.axes)
    };

    // Handling Grip poses
    if (source.gripSpace) {
      const gripPose = frame.getPose(source.gripSpace, referenceSpace);
      if (gripPose) {
        controllerData.pose = CoordinateTransformer.transformPose(gripPose.transform);
      }
    }

    // Handling Target Ray poses
    if (source.targetRaySpace) {
      const targetRayPose = frame.getPose(source.targetRaySpace, referenceSpace);
      if (targetRayPose) {
        controllerData.targetRayPose = CoordinateTransformer.transformPose(targetRayPose.transform);
      }
    }

    return controllerData;
  }
}

/**
 * UI utility class
 */
export class UIHelper {
  /**
   * Update information text
   * @param {HTMLElement} element - element
   * @param {string} text - text content
   */
  static updateInfo(element, text) {
    if (element) {
      element.textContent = text;
    }
  }

  /**
   * Shows an element.
   * @param {HTMLElement} element - The element.
   */
  static show(element) {
    if (element) {
      element.style.display = 'block';
    }
  }

  /**
   * Hides an element.
   * @param {HTMLElement} element - The element.
   */
  static hide(element) {
    if (element) {
      element.style.display = 'none';
    }
  }

  /**
   * Displays an error message.
   * @param {HTMLElement} warningElement - The warning element.
   * @param {string} message - The error message.
   */
  static showError(warningElement, message) {
    if (warningElement) {
      warningElement.textContent = message;
      this.show(warningElement);
    }
  }
}
