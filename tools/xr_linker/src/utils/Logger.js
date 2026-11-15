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
 * Enhanced logging utility module.
 * Based on Winston, it supports multiple transports, file rotation, and colorized output.
 */

import winston from 'winston';
import DailyRotateFile from 'winston-daily-rotate-file';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class Logger {
  constructor(config = {}) {
    this.config = {
      level: config.level || process.env.LOG_LEVEL || 'info',
      enableConsole: config.enableConsole !== false,
      enableFile: config.enableFile !== false,
      logDir: config.logDir || path.join(process.cwd(), 'logs'),
      maxFiles: config.maxFiles || '14d',
      maxSize: config.maxSize || '20m',
      datePattern: config.datePattern || 'YYYY-MM-DD',
      ...config
    };

    this.logger = this._createLogger();
  }

  /**
   * Creates a Winston logger instance.
   * @private
   */
  _createLogger() {
    const transports = [];

    // Console transport
    if (this.config.enableConsole) {
      transports.push(
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
            winston.format.printf(({ timestamp, level, message, ...meta }) => {
              let metaStr = '';
              if (Object.keys(meta).length > 0) {
                metaStr = ' ' + JSON.stringify(meta);
              }
              return `[${timestamp}] ${level}: ${message}${metaStr}`;
            })
          )
        })
      );
    }

    // File transport - all logs
    if (this.config.enableFile) {
      // Ensure log directory exists
      if (!fs.existsSync(this.config.logDir)) {
        fs.mkdirSync(this.config.logDir, { recursive: true });
      }

      transports.push(
        new DailyRotateFile({
          filename: path.join(this.config.logDir, 'combined-%DATE%.log'),
          datePattern: this.config.datePattern,
          maxSize: this.config.maxSize,
          maxFiles: this.config.maxFiles,
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json()
          )
        })
      );

      // File transport - error logs
      transports.push(
        new DailyRotateFile({
          level: 'error',
          filename: path.join(this.config.logDir, 'error-%DATE%.log'),
          datePattern: this.config.datePattern,
          maxSize: this.config.maxSize,
          maxFiles: this.config.maxFiles,
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json()
          )
        })
      );
    }

    return winston.createLogger({
      level: this.config.level,
      transports,
      exitOnError: false
    });
  }

  /**
   * Creates a child logger.
   * @param {string} label - The child logger label.
   * @returns {Logger} The child logger instance.
   */
  child(label) {
    const childLogger = new Logger(this.config);
    childLogger.logger = this.logger.child({ label });
    return childLogger;
  }

  /**
   * Logs a message at the error level.
   * @param {string} message - The log message.
   * @param {any} meta - Additional metadata.
   */
  error(message, meta = {}) {
    this.logger.error(message, meta);
  }

  /**
   * Logs a message at the warn level.
   * @param {string} message - The log message.
   * @param {any} meta - Additional metadata.
   */
  warn(message, meta = {}) {
    this.logger.warn(message, meta);
  }

  /**
   * Logs a message at the info level.
   * @param {string} message - The log message.
   * @param {any} meta - Additional metadata.
   */
  info(message, meta = {}) {
    this.logger.info(message, meta);
  }

  /**
   * Logs a message at the http level.
   * @param {string} message - The log message.
   * @param {any} meta - Additional metadata.
   */
  http(message, meta = {}) {
    this.logger.http(message, meta);
  }

  /**
   * Logs a message at the debug level.
   * @param {string} message - The log message.
   * @param {any} meta - Additional metadata.
   */
  debug(message, meta = {}) {
    this.logger.debug(message, meta);
  }

  /**
   * Sets the log level.
   * @param {string} level - The new log level.
   */
  setLevel(level) {
    this.logger.level = level;
    this.config.level = level;
  }

  /**
   * Gets the current log level.
   * @returns {string} The current log level.
   */
  getLevel() {
    return this.logger.level;
  }
}

// Create a default logger instance
const defaultLogger = new Logger();

export default Logger;
export { defaultLogger };
