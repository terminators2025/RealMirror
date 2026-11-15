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
 * Unit Tests for Config Module
 */
import { describe, test, expect, beforeEach } from '@jest/globals';
import config from '../../src/utils/Config.js';

describe('Config Module', () => {
  describe('Server Configuration', () => {
    test('should have default server port', () => {
      expect(config.server.port).toBeDefined();
      expect(typeof config.server.port).toBe('number');
      expect(config.server.port).toBe(8080);
    });

    test('should have default server host', () => {
      expect(config.server.host).toBeDefined();
      expect(typeof config.server.host).toBe('string');
    });
  });

  describe('HTTPS Configuration', () => {
    test('should have certificate paths', () => {
      expect(config.https.keyPath).toBeDefined();
      expect(config.https.certPath).toBeDefined();
      expect(typeof config.https.keyPath).toBe('string');
      expect(typeof config.https.certPath).toBe('string');
    });

    test('should have default certificate paths', () => {
      expect(config.https.keyPath).toContain('key.pem');
      expect(config.https.certPath).toContain('cert.pem');
    });
  });

  describe('Logging Configuration', () => {
    test('should have log level', () => {
      expect(config.logging.level).toBeDefined();
      expect(['error', 'warn', 'info', 'debug']).toContain(config.logging.level);
    });

    test('should have console logging flag', () => {
      expect(config.logging.enableConsole).toBeDefined();
      expect(typeof config.logging.enableConsole).toBe('boolean');
    });

    test('should have file logging flag', () => {
      expect(config.logging.enableFile).toBeDefined();
      expect(typeof config.logging.enableFile).toBe('boolean');
    });

    test('should have log directory', () => {
      expect(config.logging.logDir).toBeDefined();
      expect(typeof config.logging.logDir).toBe('string');
    });
  });

  describe('Storage Configuration', () => {
    test('should write the live data file atomically by default', () => {
      expect(config.storage.useTempFile).toBe(true);
    });

    test('should have data directory', () => {
      expect(config.storage.dataDir).toBeDefined();
      expect(typeof config.storage.dataDir).toBe('string');
    });

    test('should have live data file name', () => {
      expect(config.storage.liveDataFile).toBeDefined();
      expect(typeof config.storage.liveDataFile).toBe('string');
    });
  });

  describe('Socket.IO Configuration', () => {
    test('should have CORS configuration', () => {
      expect(config.socketio.cors).toBeDefined();
      expect(typeof config.socketio.cors).toBe('object');
    });

    test('should have CORS origin', () => {
      expect(config.socketio.cors.origin).toBeDefined();
    });

    test('should have CORS methods', () => {
      expect(config.socketio.cors.methods).toBeDefined();
      expect(Array.isArray(config.socketio.cors.methods)).toBe(true);
      expect(config.socketio.cors.methods).toContain('GET');
      expect(config.socketio.cors.methods).toContain('POST');
    });

    test('should have ping timeout', () => {
      expect(config.socketio.pingTimeout).toBeDefined();
      expect(typeof config.socketio.pingTimeout).toBe('number');
      expect(config.socketio.pingTimeout).toBeGreaterThan(0);
    });

    test('should have ping interval', () => {
      expect(config.socketio.pingInterval).toBeDefined();
      expect(typeof config.socketio.pingInterval).toBe('number');
      expect(config.socketio.pingInterval).toBeGreaterThan(0);
    });
  });

  describe('Configuration Structure', () => {
    test('should have all required sections', () => {
      expect(config).toHaveProperty('server');
      expect(config).toHaveProperty('https');
      expect(config).toHaveProperty('logging');
      expect(config).toHaveProperty('storage');
      expect(config).toHaveProperty('socketio');
    });
  });
});
