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
 * XR Data Collector
 * Responsible for WebXR session management and data collection
 */
import { XR_CONFIG, SOCKET_EVENTS, MESSAGES, UI_ELEMENTS } from './constants.js';
import { ControllerDataProcessor, UIHelper } from './utils.js';

export class XRDataCollector {
  /**
   * @param {Object} socket - Socket.IO client instance
   * @param {Object} latencyTracker - LatencyTracker instance (optional)
   */
  constructor(socket, latencyTracker = null) {
    this.socket = socket;
    this.xrSession = null;
    this.xrReferenceSpace = null;
    this.renderer = null;
    this.latencyTracker = latencyTracker;

    // Get UI elements
    this.elements = {
      info: document.getElementById(UI_ELEMENTS.INFO),
      button: document.getElementById(UI_ELEMENTS.ENTER_VR_BUTTON),
      warning: document.getElementById(UI_ELEMENTS.COMPATIBILITY_WARNING)
    };

    this._setupSocketListeners();
  }

  /**
   * Set up Socket listeners
   * @private
   */
  _setupSocketListeners() {
    this.socket.on(SOCKET_EVENTS.CONNECT, () => {
      console.log('[INFO] Socket connected ID:', this.socket.id);
    });

    this.socket.on(SOCKET_EVENTS.CONNECT_ERROR, (err) => {
      console.error('[ERROR] Socket connection failed:', err);
    });

    // Set up pong listener for RTT measurement
    if (this.latencyTracker) {
      this.socket.on('pong', (pongData) => {
        this.latencyTracker.processPongResponse(pongData);
      });
    }
  }

  /**
   * Initialize Three.js renderer
   * @returns {Promise<boolean>}
   */
  async initializeRenderer() {
    try {
      UIHelper.updateInfo(this.elements.info, MESSAGES.LOADING_THREE);

      // Dynamically load Three.js
      const threeModule = await import('../libs/three.module.js');
      window.THREE = threeModule;
      console.log(MESSAGES.THREE_LOADED);

      // Create a minimal renderer
      this.renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
      });
      this.renderer.setPixelRatio(window.devicePixelRatio);
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.renderer.xr.enabled = true;

      // Add to DOM but hide
      document.body.appendChild(this.renderer.domElement);
      this.renderer.domElement.style.display = 'none';

      console.log('Minimal renderer initialized');
      return true;

    } catch (error) {
      console.error('Failed to load Three.js:', error);
      UIHelper.showError(this.elements.warning, MESSAGES.LOAD_THREE_FAILED);
      return false;
    }
  }

  /**
   * Check for WebXR support
   * @returns {Promise<boolean>}
   */
  async checkXRSupport() {
    if (!navigator.xr) {
      console.error('navigator.xr does not exist');
      return false;
    }

    try {
      const supported = await navigator.xr.isSessionSupported(XR_CONFIG.SESSION_MODE);
      console.log(`'${XR_CONFIG.SESSION_MODE}' mode support status:`, supported);
      return supported;
    } catch (error) {
      console.error('Exception occurred while checking for XR support:', error);
      return false;
    }
  }

  /**
   * XR frame callback function
   * @param {number} time - Timestamp
   * @param {XRFrame} frame - XR frame
   * @private
   */
  _onXRFrame(time, frame) {
    if (!this.xrSession || !frame) {
      return;
    }

    // Check reference space
    if (!this.xrReferenceSpace) {
      console.warn('Reference space lost, trying to reacquire...');
      this.xrSession.requestReferenceSpace(XR_CONFIG.REFERENCE_SPACE)
        .then(space => {
          this.xrReferenceSpace = space;
        })
        .catch(err => {
          console.error('Could not reacquire reference space in loop:', err);
          this.xrSession.end();
        });
      return;
    }

    // Collect controller data
    const controllersData = this._collectControllerData(frame);

    // Send data
    if (controllersData.length > 0) {
      let payload = {
        timestamp: Date.now(),
        data: controllersData
      };

      // Add latency tracking metadata if tracker is available
      if (this.latencyTracker) {
        payload = this.latencyTracker.addLatencyMetadata(payload);
      }

      this.socket.emit(SOCKET_EVENTS.CONTROLLER_DATA, payload);
    }
  }

  /**
   * Collect controller data
   * @param {XRFrame} frame - XR frame
   * @returns {Array} Controller data array
   * @private
   */
  _collectControllerData(frame) {
    const controllersData = [];
    const inputSources = this.xrSession.inputSources;

    for (const source of inputSources) {
      const controllerData = ControllerDataProcessor.processController(
        source,
        frame,
        this.xrReferenceSpace
      );

      if (controllerData) {
        controllersData.push(controllerData);
      }
    }

    return controllersData;
  }

  /**
   * Start VR session
   * @returns {Promise<void>}
   */
  async startVRSession() {
    if (this.xrSession) {
      console.warn('VR session already exists');
      return;
    }

    UIHelper.updateInfo(this.elements.info, MESSAGES.REQUESTING_SESSION);

    try {
      // Request VR session
      this.xrSession = await navigator.xr.requestSession(
        XR_CONFIG.SESSION_MODE,
        {
          requiredFeatures: XR_CONFIG.REQUIRED_FEATURES,
          optionalFeatures: XR_CONFIG.OPTIONAL_FEATURES
        }
      );

      console.log('VR session established');
      UIHelper.updateInfo(this.elements.info, MESSAGES.SESSION_ESTABLISHED);
      UIHelper.hide(this.elements.button);

      // Bind session to renderer
      await this.renderer.xr.setSession(this.xrSession);
      console.log('Session successfully bound to renderer');

      // Get reference space
      this.xrReferenceSpace = await this.xrSession.requestReferenceSpace(
        XR_CONFIG.REFERENCE_SPACE
      );
      console.log(`Reference space "${XR_CONFIG.REFERENCE_SPACE}" established`);

      // Listen for session end
      this.xrSession.addEventListener('end', () => {
        this._onSessionEnd();
      });

      // Start data collection loop
      this.renderer.setAnimationLoop((time, frame) => {
        this._onXRFrame(time, frame);
      });
      console.log('Data collection loop started');

    } catch (error) {
      console.error('VR session start failed:', error);
      UIHelper.updateInfo(this.elements.info, `Error: ${error.message}`);
      UIHelper.showError(this.elements.warning, MESSAGES.SESSION_START_FAILED);

      if (this.xrSession) {
        this.xrSession.end().catch(e => console.error('Failed to end session:', e));
      }
    }
  }

  /**
   * Session end handler
   * @private
   */
  _onSessionEnd() {
    console.log('VR session ended');
    UIHelper.updateInfo(this.elements.info, MESSAGES.SESSION_ENDED);

    // Stop animation loop
    this.renderer.setAnimationLoop(null);

    // Clean up state
    this.xrSession = null;
    this.xrReferenceSpace = null;

    // Show button
    UIHelper.show(this.elements.button);
  }

  /**
   * Initialize application
   * @returns {Promise<void>}
   */
  async initialize() {
    UIHelper.updateInfo(this.elements.info, MESSAGES.INITIALIZING);

    // Initialize renderer
    if (!(await this.initializeRenderer())) {
      UIHelper.updateInfo(this.elements.info, MESSAGES.RENDERER_INIT_FAILED);
      return;
    }

    // Check XR support
    const supported = await this.checkXRSupport();

    if (supported) {
      UIHelper.updateInfo(this.elements.info, MESSAGES.CLICK_TO_ENTER);
      UIHelper.show(this.elements.button);

      // Bind button event
      this.elements.button.addEventListener('click', () => {
        this.startVRSession();
      });
    } else {
      // If XR is not supported, update info and show warning
      UIHelper.updateInfo(this.elements.info, MESSAGES.NO_XR_SUPPORT);
      UIHelper.hide(this.elements.button);
      UIHelper.showError(this.elements.warning, MESSAGES.NO_XR_SUPPORT);
    }
  }
}
