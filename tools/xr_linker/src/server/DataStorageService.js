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
 * Data Storage Service
 * Responsible for writing received Pico data to the file system.
 */
import fs from 'fs';
import path from 'path';

class DataStorageService {
  /**
   * @param {Object} config - Storage configuration.
   * @param {Object} logger - Logger instance.
   */
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.latestFrame = null;
    this.isWriting = false;

    // Initialize storage directory
    this._initializeStorage();
  }

  /**
   * Initializes the storage directory.
   * @private
   */
  _initializeStorage() {
    try {
      const dataDir = this.config.dataDir;
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
        this.logger.info(`Data directory created: ${dataDir}`);
      }
    } catch (error) {
      this.logger.error('Failed to initialize storage directory', { error: error.message });
      throw error;
    }
  }

  /**
   * Gets the full path of the data file.
   * @returns {string}
   */
  getDataFilePath() {
    return path.join(this.config.dataDir, this.config.liveDataFile);
  }

  /**
   * Gets the full path of the temporary file.
   * @returns {string}
   */
  getTempFilePath() {
    return `${this.getDataFilePath()}.tmp`;
  }

  /**
   * Validates the data format.
   * @param {Object} data - The data to validate.
   * @returns {boolean}
   * @private
   */
  _validateData(data) {
    if (!data || typeof data !== 'object') {
      return false;
    }

    if (!data.timestamp || typeof data.timestamp !== 'number') {
      return false;
    }

    if (!Array.isArray(data.data)) {
      return false;
    }

    return true;
  }

  /**
   * Updates the latest frame data.
   * @param {Object} frameData - Frame data {timestamp, data}.
   */
  updateLatestFrame(frameData) {
    if (!this._validateData(frameData)) {
      this.logger.warn('Received invalid data format', { frameData });
      return false;
    }

    this.latestFrame = {
      timestamp: frameData.timestamp,
      data: frameData.data
    };

    this.logger.debug('Latest frame data updated', {
      timestamp: this.latestFrame.timestamp,
      controllerCount: this.latestFrame.data.length
    });

    return true;
  }

  /**
   * Writes the latest data to a file (atomic write).
   * @param {Object} dataToWrite - Optional data to write (defaults to latestFrame).
   * @returns {Promise<boolean>}
   */
  async writeLatestData(dataToWrite = null) {
    const dataSource = dataToWrite || this.latestFrame;

    if (!dataSource) {
      this.logger.warn('No data available to write');
      return false;
    }

    if (this.isWriting) {
      this.logger.debug('Currently writing, skipping this write operation');
      return false;
    }

    this.isWriting = true;

    try {
      const dataPath = this.getDataFilePath();
      const tempPath = this.getTempFilePath();
      const jsonData = JSON.stringify(dataSource, null, 2);

      if (this.config.useTempFile) {
        // Use a temporary file for atomic write
        await fs.promises.writeFile(tempPath, jsonData, 'utf8');
        await fs.promises.rename(tempPath, dataPath);
      } else {
        // Direct write
        await fs.promises.writeFile(dataPath, jsonData, 'utf8');
      }

      this.logger.debug('Data written to file', { path: dataPath });
      return true;

    } catch (error) {
      this.logger.error('Failed to write data file', {
        error: error.message,
        stack: error.stack
      });
      return false;
    } finally {
      this.isWriting = false;
    }
  }

  /**
   * Gets the latest frame data.
   * @returns {Object|null}
   */
  getLatestFrame() {
    return this.latestFrame;
  }

  /**
   * Clears the latest frame data.
   */
  clearLatestFrame() {
    this.latestFrame = null;
    this.logger.info('Latest frame data cleared');
  }

  /**
   * Deletes the data file.
   * @returns {Promise<boolean>}
   */
  async deleteDataFile() {
    try {
      const dataPath = this.getDataFilePath();
      if (fs.existsSync(dataPath)) {
        await fs.promises.unlink(dataPath);
        this.logger.info('Data file deleted', { path: dataPath });
        return true;
      }
      return false;
    } catch (error) {
      this.logger.error('Failed to delete data file', { error: error.message });
      return false;
    }
  }
}

export default DataStorageService;
