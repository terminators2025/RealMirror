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
 * Configuration management module
 * Supports environment variables and default configurations
 */

import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config();

/**
 * Parses a boolean environment variable.
 * @param {string} value - The environment variable value.
 * @param {boolean} defaultValue - The default value.
 * @returns {boolean}
 */
function parseBoolean(value, defaultValue) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  return value === 'true' || value === '1';
}

/**
 * Application configuration
 */
const config = {
  // Environment
  env: process.env.NODE_ENV || 'development',

  // Server configuration
  server: {
    port: parseInt(process.env.PORT || '8080', 10),
    host: process.env.HOST || '0.0.0.0'
  },

  // HTTPS certificate configuration
  https: {
    keyPath: process.env.HTTPS_KEY_PATH || 'cert/key.pem',
    certPath: process.env.HTTPS_CERT_PATH || 'cert/cert.pem'
  },

  // Socket.IO configuration
  socketio: {
    cors: {
      origin: process.env.SOCKET_CORS_ORIGIN || '*',
      methods: ['GET', 'POST']
    },
    pingTimeout: parseInt(process.env.SOCKET_PING_TIMEOUT || '60000', 10),
    pingInterval: parseInt(process.env.SOCKET_PING_INTERVAL || '25000', 10)
  },

  // Data storage configuration
  // The live data file is the transport to the simulator, not an archive:
  // one fixed file always holds the most recent frame. `useTempFile` writes
  // through `<file>.tmp` and renames, so readers never observe a partial frame.
  storage: {
    dataDir: process.env.DATA_STORAGE_DIR || 'data',
    liveDataFile: 'live_pico_data.json',
    useTempFile: true
  },

  // Logging configuration
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    enableConsole: parseBoolean(process.env.LOG_ENABLE_CONSOLE, true),
    enableFile: parseBoolean(process.env.LOG_ENABLE_FILE, true),
    logDir: process.env.LOG_DIR || './logs',
    maxFiles: process.env.LOG_MAX_FILES || '14d',
    maxSize: process.env.LOG_MAX_SIZE || '20m'
  }
};

export default config;
