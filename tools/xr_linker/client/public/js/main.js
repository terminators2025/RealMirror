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
 * WebXR client main entry point
 * Responsible for initializing and starting the XR data collection application
 */
import { XRDataCollector } from './XRDataCollector.js';

(async () => {
  // Initialize Socket.IO connection
  const socket = io();

  // Initialize LatencyTracker (load from separate script tag)
  let latencyTracker = null;
  if (typeof LatencyTracker !== 'undefined') {
    latencyTracker = new LatencyTracker();
    latencyTracker.setEnabled(true);
    console.log('LatencyTracker initialized and enabled');
  } else {
    console.warn('LatencyTracker not available - latency tracking disabled');
  }

  // Create XR data collector with latency tracker
  const collector = new XRDataCollector(socket, latencyTracker);

  // Initialize the application
  await collector.initialize();

  // Optional: Send periodic ping for RTT measurement
  if (latencyTracker) {
    setInterval(() => {
      const pingPacket = latencyTracker.generatePingPacket();
      socket.emit('ping', pingPacket);
    }, 5000); // Ping every 5 seconds
  }
})();
