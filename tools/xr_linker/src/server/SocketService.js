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
 * Socket.IO Service
 * Responsible for handling WebSocket connections and data transmission.
 */
import { Server } from 'socket.io';
import LatencyMonitor from './LatencyMonitor.js';

class SocketService {
  /**
   * @param {Object} httpsServer - HTTPS server instance.
   * @param {Object} config - Socket.IO configuration.
   * @param {Object} logger - Logger instance.
   * @param {Object} dataStorage - Data storage service instance.
   */
  constructor(httpsServer, config, logger, dataStorage) {
    this.logger = logger;
    this.dataStorage = dataStorage;

    // Latency monitoring settings are ours, not Socket.IO's, so keep them out
    // of the options handed to the engine.
    const { latency: latencyOptions, ...socketOptions } = config || {};

    // Initialize LatencyMonitor
    this.latencyMonitor = new LatencyMonitor({
      enabled: true,
      logToFile: true,
      ...(latencyOptions || {})
    });
    this.connectedClients = new Map();

    // Data reception statistics
    this.dataStats = {
      totalReceived: 0,
      lastLogTime: Date.now(),
      logInterval: 5000, // Log statistics every 5 seconds
      receivedSinceLastLog: 0
    };

    // Initialize Socket.IO server
    this.io = new Server(httpsServer, socketOptions);
    this._setupEventHandlers();
  }

  /**
   * Sets up event handlers.
   * @private
   */
  _setupEventHandlers() {
    this.io.on('connection', (socket) => this._handleConnection(socket));
  }

  /**
   * Handles client connection.
   * @param {Object} socket - Socket instance.
   * @private
   */
  _handleConnection(socket) {
    const clientInfo = {
      id: socket.id,
      connectedAt: new Date().toISOString(),
      address: socket.handshake.address
    };

    this.connectedClients.set(socket.id, clientInfo);
    this.logger.info('Client connected', {
      clientId: socket.id,
      totalClients: this.connectedClients.size
    });

    // Register event listeners
    socket.on('controller-data', (data) => this._handleControllerData(socket, data));
    socket.on('ping', (data) => this._handlePing(socket, data));
    socket.on('disconnect', (reason) => this._handleDisconnect(socket, reason));
    socket.on('error', (error) => this._handleError(socket, error));
  }

  /**
   * Handles controller data.
   * @param {Object} socket - Socket instance.
   * @param {Object} picoData - Pico device data.
   * @private
   */
  async _handleControllerData(socket, picoData) {
    try {
      // Add server receive timestamp via LatencyMonitor
      const augmentedData = this.latencyMonitor.onPacketReceived(picoData);

      // Update statistics
      this.dataStats.totalReceived++;
      this.dataStats.receivedSinceLastLog++;

      // Log statistics at regular intervals
      const now = Date.now();
      if (now - this.dataStats.lastLogTime >= this.dataStats.logInterval) {
        const fps = (this.dataStats.receivedSinceLastLog / (now - this.dataStats.lastLogTime) * 1000).toFixed(1);

        this.logger.info('Controller data statistics', {
          clientId: socket.id,
          totalReceived: this.dataStats.totalReceived,
          recentCount: this.dataStats.receivedSinceLastLog,
          fps: `${fps} frames/sec`,
          controllerCount: augmentedData.data?.length || 0
        });

        // Reset statistics
        this.dataStats.lastLogTime = now;
        this.dataStats.receivedSinceLastLog = 0;

        // Log latency statistics
        this.latencyMonitor.logStats();
      }

      // Update latest frame data with augmented data
      const updated = this.dataStorage.updateLatestFrame(augmentedData);

      if (updated) {
        // Add forward timestamp before writing
        const finalData = this.latencyMonitor.onPacketForwarded(augmentedData);

        // Write to file with latency metadata
        await this.dataStorage.writeLatestData(finalData);
      }

    } catch (error) {
      this.logger.error('Error processing controller data', {
        clientId: socket.id,
        error: error.message,
        stack: error.stack
      });
    }
  }

  /**
   * Handles ping requests for RTT measurement.
   * @param {Object} socket - Socket instance.
   * @param {Object} pingData - Ping data.
   * @private
   */
  _handlePing(socket, pingData) {
    try {
      const pongData = this.latencyMonitor.handlePing(pingData);
      socket.emit('pong', pongData);
    } catch (error) {
      this.logger.error('Error handling ping', {
        clientId: socket.id,
        error: error.message
      });
    }
  }

  /**
   * Handles client disconnection.
   * @param {Object} socket - Socket instance.
   * @param {string} reason - Disconnection reason.
   * @private
   */
  _handleDisconnect(socket, reason) {
    this.connectedClients.delete(socket.id);

    this.logger.info('Client disconnected', {
      clientId: socket.id,
      reason: reason,
      remainingClients: this.connectedClients.size
    });

    // If no clients are connected, reset statistics and clear data
    if (this.connectedClients.size === 0) {
      this.logger.info('All clients disconnected');

      // Reset data statistics
      this.dataStats.totalReceived = 0;
      this.dataStats.lastLogTime = Date.now();
      this.dataStats.receivedSinceLastLog = 0;

      // Optional: Clear latest frame data or delete file
      // this.dataStorage.clearLatestFrame();
      // this.dataStorage.deleteDataFile();
    }
  }

  /**
   * Handles Socket errors.
   * @param {Object} socket - Socket instance.
   * @param {Error} error - Error object.
   * @private
   */
  _handleError(socket, error) {
    this.logger.error('Socket error', {
      clientId: socket.id,
      error: error.message,
      stack: error.stack
    });
  }

  /**
   * Gets the number of connected clients.
   * @returns {number}
   */
  getConnectedClientCount() {
    return this.connectedClients.size;
  }

  /**
   * Gets information about all connected clients.
   * @returns {Array}
   */
  getConnectedClients() {
    return Array.from(this.connectedClients.values());
  }

  /**
   * Broadcasts a message to all connected clients.
   * @param {string} event - Event name.
   * @param {any} data - Data.
   */
  broadcast(event, data) {
    this.io.emit(event, data);
    this.logger.debug('Broadcasting message', { event, clientCount: this.connectedClients.size });
  }

  /**
   * Gets latency statistics from the monitor.
   * @returns {Object}
   */
  getLatencyStats() {
    return this.latencyMonitor.getStats();
  }

  /**
   * Closes the Socket.IO server.
   * @returns {Promise<void>}
   */
  async close() {
    // Shutdown latency monitor before closing
    this.latencyMonitor.shutdown();

    return new Promise((resolve) => {
      this.io.close(() => {
        this.logger.info('Socket.IO server closed');
        resolve();
      });
    });
  }
}

export default SocketService;
