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
 * Unit Tests for DataStorageService
 */
import { describe, test, expect, beforeEach, afterEach, jest } from '@jest/globals';
import DataStorageService from '../../src/server/DataStorageService.js';
import fs from 'fs';
import path from 'path';

describe('DataStorageService', () => {
  let dataStorage;
  let mockLogger;
  const testDataDir = './test_data';
  const testDataFile = 'test_data.json';

  beforeEach(() => {
    // Create mock logger
    mockLogger = {
      info: jest.fn(),
      error: jest.fn(),
      warn: jest.fn(),
      debug: jest.fn()
    };

    // Create test config
    const config = {
      dataDir: testDataDir,
      liveDataFile: testDataFile
    };

    dataStorage = new DataStorageService(config, mockLogger);
  });

  afterEach(() => {
    // Clean up test data directory
    if (fs.existsSync(testDataDir)) {
      fs.rmSync(testDataDir, { recursive: true, force: true });
    }
  });

  describe('Initialization', () => {
    test('should create instance with valid config', () => {
      expect(dataStorage).toBeDefined();
      expect(dataStorage.config).toBeDefined();
      expect(dataStorage.logger).toBe(mockLogger);
    });

    test('should create data directory if not exists', () => {
      expect(fs.existsSync(testDataDir)).toBe(true);
    });

    test('should have correct data file path', () => {
      const expectedPath = path.join(testDataDir, testDataFile);
      expect(dataStorage.getDataFilePath()).toBe(expectedPath);
    });

    test('should replace the live file without leaving a temp file behind', async () => {
      const atomicStorage = new DataStorageService(
        { dataDir: testDataDir, liveDataFile: testDataFile, useTempFile: true },
        mockLogger
      );

      atomicStorage.updateLatestFrame({
        timestamp: Date.now(),
        data: [{ id: 'atomic-frame' }]
      });

      expect(await atomicStorage.writeLatestData()).toBe(true);
      expect(fs.existsSync(atomicStorage.getDataFilePath())).toBe(true);
      expect(fs.existsSync(atomicStorage.getTempFilePath())).toBe(false);
    });
  });

  describe('Data Validation and Update', () => {
    test('should update latest frame with correct data format', () => {
      const validData = {
        timestamp: Date.now(),
        data: [
          {
            id: 'left',
            handedness: 'left',
            buttons: [{ pressed: false, touched: false, value: 0 }],
            axes: [0, 0],
            pose: {
              position: [0, 0, 0],
              orientation: [0, 0, 0, 1]
            }
          }
        ]
      };

      const result = dataStorage.updateLatestFrame(validData);
      expect(result).toBe(true);
      expect(dataStorage.getLatestFrame()).toEqual(validData);
    });

    test('should reject data without timestamp', () => {
      const invalidData = {
        data: []
      };

      const result = dataStorage.updateLatestFrame(invalidData);
      expect(result).toBe(false);
    });

    test('should reject data without data array', () => {
      const invalidData = {
        timestamp: Date.now()
      };

      const result = dataStorage.updateLatestFrame(invalidData);
      expect(result).toBe(false);
    });

    test('should reject null data', () => {
      const result = dataStorage.updateLatestFrame(null);
      expect(result).toBe(false);
    });

    test('should reject undefined data', () => {
      const result = dataStorage.updateLatestFrame(undefined);
      expect(result).toBe(false);
    });

    test('should reject data with invalid timestamp', () => {
      const invalidData = {
        timestamp: 'invalid',
        data: []
      };

      const result = dataStorage.updateLatestFrame(invalidData);
      expect(result).toBe(false);
    });

    test('should reject data with non-array data field', () => {
      const invalidData = {
        timestamp: Date.now(),
        data: 'not an array'
      };

      const result = dataStorage.updateLatestFrame(invalidData);
      expect(result).toBe(false);
    });
  });

  describe('Data Storage', () => {
    test('should write valid data', async () => {
      const testData = {
        timestamp: Date.now(),
        data: [
          {
            id: 'left',
            handedness: 'left',
            buttons: [],
            axes: [],
            pose: { position: [0, 0, 0], orientation: [0, 0, 0, 1] }
          }
        ]
      };

      dataStorage.updateLatestFrame(testData);
      const result = await dataStorage.writeLatestData();

      expect(result).toBe(true);
      const filePath = dataStorage.getDataFilePath();
      expect(fs.existsSync(filePath)).toBe(true);

      const savedData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      expect(savedData).toEqual(testData);
    });

    test('should not write when no data available', async () => {
      const result = await dataStorage.writeLatestData();
      expect(result).toBe(false);
      expect(mockLogger.warn).toHaveBeenCalled();
    });

    test('should handle concurrent write attempts', async () => {
      const testData = {
        timestamp: Date.now(),
        data: [{ id: 'test1' }]
      };

      dataStorage.updateLatestFrame(testData);

      // Attempt concurrent writes
      const results = await Promise.all([
        dataStorage.writeLatestData(),
        dataStorage.writeLatestData()
      ]);

      // At least one should succeed
      expect(results.includes(true)).toBe(true);
    });
  });

  describe('Error Handling', () => {
    test('should handle file system errors gracefully', async () => {
      // Create a read-only directory
      const readOnlyDir = './readonly_test';
      if (!fs.existsSync(readOnlyDir)) {
        fs.mkdirSync(readOnlyDir);
      }
      fs.chmodSync(readOnlyDir, 0o444);

      const errorConfig = {
        dataDir: readOnlyDir,
        liveDataFile: 'test.json'
      };

      const errorStorage = new DataStorageService(errorConfig, mockLogger);

      const testData = {
        timestamp: Date.now(),
        data: []
      };

      errorStorage.updateLatestFrame(testData);
      const result = await errorStorage.writeLatestData();

      expect(result).toBe(false);
      expect(mockLogger.error).toHaveBeenCalled();

      // Cleanup
      fs.chmodSync(readOnlyDir, 0o755);
      fs.rmSync(readOnlyDir, { recursive: true, force: true });
    });
  });

  describe('Utility Methods', () => {
    test('should return correct data file path', () => {
      const expectedPath = path.join(testDataDir, testDataFile);
      expect(dataStorage.getDataFilePath()).toBe(expectedPath);
    });

    test('should get and clear latest frame', () => {
      const testData = {
        timestamp: Date.now(),
        data: []
      };

      dataStorage.updateLatestFrame(testData);
      expect(dataStorage.getLatestFrame()).toEqual(testData);

      dataStorage.clearLatestFrame();
      expect(dataStorage.getLatestFrame()).toBeNull();
    });

    test('should delete data file', async () => {
      const testData = {
        timestamp: Date.now(),
        data: []
      };

      dataStorage.updateLatestFrame(testData);
      await dataStorage.writeLatestData();

      const filePath = dataStorage.getDataFilePath();
      expect(fs.existsSync(filePath)).toBe(true);

      const result = await dataStorage.deleteDataFile();
      expect(result).toBe(true);
      expect(fs.existsSync(filePath)).toBe(false);
    });
  });
});
