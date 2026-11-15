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
 * Client-side constant configuration
 */

// Data precision configuration
export const PRECISION = {
  POSITION: 3,      // Position data is kept to 3 decimal places
  ORIENTATION: 3,   // Direction data is kept to 3 decimal places
  BUTTON_VALUE: 2,  // The button value is kept to 2 decimal places
  AXES: 2          // The joystick axis value is kept to 2 decimal places
};

// UI element ID
export const UI_ELEMENTS = {
  INFO: 'info',
  ENTER_VR_BUTTON: 'enter-vr',
  COMPATIBILITY_WARNING: 'compatibility-warning',
  LOADING_BAR: 'loading-bar',
  PROGRESS: 'progress'
};

// WebXR configuration
export const XR_CONFIG = {
  SESSION_MODE: 'immersive-vr',
  REQUIRED_FEATURES: ['local-floor'],
  OPTIONAL_FEATURES: ['hand-tracking', 'bounded-floor'],
  REFERENCE_SPACE: 'local-floor'
};

// Socket.IO event name
export const SOCKET_EVENTS = {
  CONNECT: 'connect',
  CONNECT_ERROR: 'connect_error',
  DISCONNECT: 'disconnect',
  CONTROLLER_DATA: 'controller-data'
};

// Message text
export const MESSAGES = {
  INITIALIZING: 'Initializing...',
  LOADING_THREE: 'Loading Three.js library...',
  THREE_LOADED: 'Three.js library loaded successfully',
  RENDERER_INIT_FAILED: 'Renderer initialization failed and cannot continue',
  CLICK_TO_ENTER: 'Please click the button to enter VR mode',
  REQUESTING_SESSION: 'Requesting VR session...',
  SESSION_ESTABLISHED: 'VR session has been established. Data is being collected...',
  SESSION_ENDED: 'The VR session has ended. Click the button to restart',
  NO_XR_SUPPORT: 'Your browser or device does not support WebXR immersive-vr mode',
  LOAD_THREE_FAILED: 'Error: Failed to load Three.js core library',
  SESSION_START_FAILED: 'Failed to start VR session, please check the device'
};
