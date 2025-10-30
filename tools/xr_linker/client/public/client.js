/**
 * WebXR Client Main Entry
 * Responsible for initializing and starting the XR data collection application
 */
import { XRDataCollector } from './js/XRDataCollector.js';

(async () => {
  // Initialize Socket.IO connection
  const socket = io();

  // Create XR data collector
  const collector = new XRDataCollector(socket);

  // Initialize application
  await collector.initialize();
})();