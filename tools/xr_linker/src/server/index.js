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
 * XR Linker - Main Server
 * Used to receive remote operation data from XR devices like Pico and save it in real-time.
 */
import fs from 'fs';
import https from 'https';
import path from 'path';
import express from 'express';
import { fileURLToPath } from 'url';

// Import configuration and services
import config from '../utils/Config.js';
import Logger from '../utils/Logger.js';
import DataStorageService from './DataStorageService.js';
import SocketService from './SocketService.js';

// Get __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Main application class
 */
class XRLinkerServer {
  constructor() {
    this.logger = new Logger(config.logging);
    this.app = express();
    this.httpsServer = null;
    this.dataStorage = null;
    this.socketService = null;
  }

  /**
   * Loads HTTPS certificates.
   * @returns {Object} Certificate options.
   * @private
   */
  _loadCertificates() {
    try {
      const keyPath = path.join(__dirname, '../../', config.https.keyPath);
      const certPath = path.join(__dirname, '../../', config.https.certPath);

      if (!fs.existsSync(keyPath) || !fs.existsSync(certPath)) {
        throw new Error(`Certificate file not found: ${keyPath} or ${certPath}`);
      }

      return {
        key: fs.readFileSync(keyPath),
        cert: fs.readFileSync(certPath)
      };
    } catch (error) {
      this.logger.error('Failed to load HTTPS certificates', { error: error.message });
      throw error;
    }
  }

  /**
   * Configures Express middleware.
   * @private
   */
  _setupMiddleware() {
    // Static file service
    const publicPath = path.join(__dirname, '../../client/public');
    this.app.use(express.static(publicPath));
    this.logger.info('Static file directory configured', { path: publicPath });

    // Other middleware can be added, such as logging, CORS, etc.
  }

  /**
   * Initializes services.
   * @private
   */
  _initializeServices() {
    // Create HTTPS server
    const certOptions = this._loadCertificates();
    this.httpsServer = https.createServer(certOptions, this.app);
    this.logger.info('HTTPS server created');

    // Initialize data storage service
    this.dataStorage = new DataStorageService(config.storage, this.logger);
    this.logger.info('Data storage service initialized', {
      dataPath: this.dataStorage.getDataFilePath()
    });

    // Initialize Socket.IO service
    this.socketService = new SocketService(
      this.httpsServer,
      config.socketio,
      this.logger,
      this.dataStorage
    );
    this.logger.info('Socket.IO service initialized');
  }

  /**
   * Sets up graceful shutdown.
   * @private
   */
  _setupGracefulShutdown() {
    const shutdown = async (signal) => {
      this.logger.info(`Received ${signal} signal, starting graceful shutdown...`);

      try {
        // Close Socket.IO service
        if (this.socketService) {
          await this.socketService.close();
        }

        // Close HTTPS server
        if (this.httpsServer) {
          await new Promise((resolve) => {
            this.httpsServer.close(() => {
              this.logger.info('HTTPS server closed');
              resolve();
            });
          });
        }

        this.logger.info('Server gracefully shut down');
        process.exit(0);
      } catch (error) {
        this.logger.error('Error while shutting down server', { error: error.message });
        process.exit(1);
      }
    };

    // Listen for termination signals
    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));

    // Listen for uncaught exceptions
    process.on('uncaughtException', (error) => {
      this.logger.error('Uncaught exception', {
        error: error.message,
        stack: error.stack
      });
      shutdown('uncaughtException');
    });

    process.on('unhandledRejection', (reason, promise) => {
      this.logger.error('Unhandled Promise rejection', {
        reason: reason,
        promise: promise
      });
    });
  }

  /**
   * Starts the server.
   */
  async start() {
    try {
      this.logger.info('Starting XR Linker server...');

      // Configure middleware
      this._setupMiddleware();

      // Initialize services
      this._initializeServices();

      // Set up graceful shutdown
      this._setupGracefulShutdown();

      // Start HTTPS server
      const { port, host } = config.server;
      await new Promise((resolve, reject) => {
        this.httpsServer.listen(port, host, (error) => {
          if (error) {
            reject(error);
          }
          else {
            resolve();
          }
        });
      });

      this.logger.info('Server started successfully', {
        port: port,
        host: host,
        url: `https://${host}:${port}`,
        dataFile: this.dataStorage.getDataFilePath()
      });

      console.log('\n========================================');
      console.log('  XR Linker Server Started');
      console.log('========================================');
      console.log(`  Local Access: https://localhost:${port}`);
      console.log(`  Network Access: https://<YOUR_IP>:${port}`);
      console.log(`  Data File: ${this.dataStorage.getDataFilePath()}`);
      console.log('========================================\n');

    } catch (error) {
      this.logger.error('Failed to start server', {
        error: error.message,
        stack: error.stack
      });
      process.exit(1);
    }
  }
}

// Create and start the server
const server = new XRLinkerServer();
server.start();
