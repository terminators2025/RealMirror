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
 * Unit Tests for SocketService
 */
import { describe, test, expect, beforeEach, jest } from '@jest/globals';
import SocketService from '../../src/server/SocketService.js';

describe('SocketService', () => {
  let socketService;
  let mockServer;
  let mockLogger;
  let mockDataStorage;

  beforeEach(() => {
    // Create mock logger
    mockLogger = {
      info: jest.fn(),
      error: jest.fn(),
      warn: jest.fn(),
      debug: jest.fn()
    };

    // Create mock data storage
    mockDataStorage = {
      updateLatestFrame: jest.fn().mockReturnValue(true),
      writeLatestData: jest.fn().mockResolvedValue(true),
      clearLatestFrame: jest.fn(),
      deleteDataFile: jest.fn().mockResolvedValue(true)
    };

    // Create mock HTTP server
    mockServer = {};

    const config = {
      cors: {
        origin: '*',
        methods: ['GET', 'POST']
      },
      pingTimeout: 10000,
      pingInterval: 5000,
      // Keep unit tests from writing latency logs into the repository
      latency: {
        enabled: false,
        logToFile: false
      }
    };

    socketService = new SocketService(mockServer, config, mockLogger, mockDataStorage);
  });

  describe('Initialization', () => {
    test('should create instance with valid parameters', () => {
      expect(socketService).toBeDefined();
      expect(socketService.dataStorage).toBe(mockDataStorage);
      expect(socketService.logger).toBe(mockLogger);
    });

    test('should initialize connected clients map', () => {
      expect(socketService.connectedClients).toBeDefined();
      expect(socketService.connectedClients.size).toBe(0);
    });

    test('should initialize data statistics', () => {
      expect(socketService.dataStats).toBeDefined();
      expect(socketService.dataStats.totalReceived).toBe(0);
    });

    test('should initialize Socket.IO server', () => {
      expect(socketService.io).toBeDefined();
    });

    test('should honour latency settings supplied by the caller', () => {
      expect(socketService.latencyMonitor.enabled).toBe(false);
      expect(socketService.latencyMonitor.logToFile).toBe(false);
    });
  });

  describe('Client Management', () => {
    test('should get connected client count', () => {
      expect(socketService.getConnectedClientCount()).toBe(0);
    });

    test('should get connected clients list', () => {
      const clients = socketService.getConnectedClients();
      expect(Array.isArray(clients)).toBe(true);
      expect(clients.length).toBe(0);
    });
  });

  describe('Broadcasting', () => {
    test('should broadcast to all clients', () => {
      const spy = jest.spyOn(socketService.io, 'emit');
      socketService.broadcast('test-event', { message: 'test' });
      expect(spy).toHaveBeenCalledWith('test-event', { message: 'test' });
    });
  });

  describe('Server Lifecycle', () => {
    test('should have close method', () => {
      expect(typeof socketService.close).toBe('function');
    });
  });
});
