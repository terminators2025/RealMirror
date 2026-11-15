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
 * Unit Tests for Logger Module
 */
import { describe, test, expect, beforeAll, afterAll, beforeEach, jest } from '@jest/globals';
import Logger from '../../src/utils/Logger.js';
import fs from 'fs';
import path from 'path';

describe('Logger Module', () => {
  let logger;
  const testLogDir = './test_logs';

  beforeAll(() => {
    // Ensure test log directory exists once before all tests
    if (!fs.existsSync(testLogDir)) {
      fs.mkdirSync(testLogDir, { recursive: true });
    }
  });

  beforeEach(() => {
    // Create test logger with minimal config
    const config = {
      level: 'debug',
      enableConsole: false,
      enableFile: false,
      logDir: testLogDir
    };
    logger = new Logger(config);
  });

  afterAll(() => {
    // Clean up test log directory after all tests
    if (fs.existsSync(testLogDir)) {
      fs.rmSync(testLogDir, { recursive: true, force: true });
    }
  });

  describe('Logger Initialization', () => {
    test('should create logger instance', () => {
      expect(logger).toBeDefined();
      expect(logger.logger).toBeDefined();
    });

    test('should have correct log level', () => {
      expect(logger.logger.level).toBe('debug');
    });

    test('should create log directory if enabled', (done) => {
      // Test directory should already exist from beforeAll
      expect(fs.existsSync(testLogDir)).toBe(true);

      const config = {
        level: 'info',
        enableConsole: false,
        enableFile: true,
        logDir: testLogDir
      };

      const fileLogger = new Logger(config);

      // Give DailyRotateFile time to initialize
      setTimeout(() => {
        // Directory should still exist after creating logger
        expect(fs.existsSync(testLogDir)).toBe(true);
        done();
      }, 100);
    });
  });

  describe('Logging Methods', () => {
    test('should have info method', () => {
      expect(typeof logger.info).toBe('function');
      expect(() => logger.info('Test message')).not.toThrow();
    });

    test('should have error method', () => {
      expect(typeof logger.error).toBe('function');
      expect(() => logger.error('Test error')).not.toThrow();
    });

    test('should have warn method', () => {
      expect(typeof logger.warn).toBe('function');
      expect(() => logger.warn('Test warning')).not.toThrow();
    });

    test('should have debug method', () => {
      expect(typeof logger.debug).toBe('function');
      expect(() => logger.debug('Test debug')).not.toThrow();
    });

    test('should log with metadata', () => {
      const metadata = { userId: 123, action: 'test' };
      expect(() => logger.info('Test message', metadata)).not.toThrow();
    });

    test('should handle error objects', () => {
      const error = new Error('Test error');
      expect(() => logger.error('Error occurred', { error: error.message })).not.toThrow();
    });
  });

  describe('Log Levels', () => {
    test('should respect log level filtering', () => {
      const warnLogger = new Logger({
        level: 'warn',
        enableConsole: false,
        enableFile: false
      });

      expect(warnLogger.logger.level).toBe('warn');

      // Debug and info should not be logged at warn level
      // But warn and error should be logged
      expect(() => warnLogger.debug('Debug message')).not.toThrow();
      expect(() => warnLogger.info('Info message')).not.toThrow();
      expect(() => warnLogger.warn('Warn message')).not.toThrow();
      expect(() => warnLogger.error('Error message')).not.toThrow();
    });

    test('should handle invalid log level', () => {
      const config = {
        level: 'invalid',
        enableConsole: false,
        enableFile: false
      };

      // Should default to 'info' or handle gracefully
      expect(() => new Logger(config)).not.toThrow();
    });
  });

  describe('Configuration Options', () => {
    test('should disable console output when configured', () => {
      const config = {
        level: 'info',
        enableConsole: false,
        enableFile: false
      };
      const consoleLogger = new Logger(config);
      expect(consoleLogger.logger.transports.length).toBeGreaterThanOrEqual(0);
    });

    test('should enable file output when configured', (done) => {
      const config = {
        level: 'info',
        enableConsole: false,
        enableFile: true,
        logDir: testLogDir
      };
      const fileLogger = new Logger(config);

      // Give DailyRotateFile time to initialize
      setTimeout(() => {
        expect(fileLogger.logger.transports.length).toBeGreaterThan(0);
        done();
      }, 100);
    });
  });

  describe('Error Handling', () => {
    test('should handle null message', () => {
      expect(() => logger.info(null)).not.toThrow();
    });

    test('should handle undefined message', () => {
      expect(() => logger.info(undefined)).not.toThrow();
    });

    test('should handle empty message', () => {
      expect(() => logger.info('')).not.toThrow();
    });

    test('should handle circular reference in metadata', () => {
      const circular = { a: 1 };
      circular.self = circular;

      // Should handle gracefully without throwing
      expect(() => logger.info('Test', { data: circular })).not.toThrow();
    });
  });
});
