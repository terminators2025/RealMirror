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
 * Integration Tests for XR Linker Server
 * Tests the complete server startup, HTTPS, Socket.IO, and data flow
 */
import { describe, test, expect, beforeAll, afterAll, jest } from '@jest/globals';
import https from 'https';
import fs from 'fs';
import { io as ioClient } from 'socket.io-client';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import the server components
import Logger from '../../src/utils/Logger.js';
import DataStorageService from '../../src/server/DataStorageService.js';
import SocketService from '../../src/server/SocketService.js';
import { createTestCertificate } from '../helpers/testCertificate.js';

// Mint the suite's own certificate at collection time so the whole block can be
// skipped when openssl is unavailable, instead of failing on a missing key.
const testCertificate = createTestCertificate();
const describeWithTls = testCertificate.ok ? describe : describe.skip;

if (!testCertificate.ok) {
  console.warn(`Skipping XR Linker HTTPS integration tests: ${testCertificate.reason}`);
}

describeWithTls('XR Linker Server Integration Tests', () => {
  let httpsServer;
  let socketService;
  let dataStorage;
  let logger;
  let client;

  const testPort = 8444;
  const testDataDir = './test_integration_data';
  const testLogDir = './test_integration_logs';

  beforeAll((done) => {
    // Create test directories
    if (!fs.existsSync(testDataDir)) {
      fs.mkdirSync(testDataDir, { recursive: true });
    }
    if (!fs.existsSync(testLogDir)) {
      fs.mkdirSync(testLogDir, { recursive: true });
    }

    // Initialize logger
    logger = new Logger({
      level: 'error',
      enableConsole: false,
      enableFile: false,
      logDir: testLogDir
    });

    // Initialize data storage
    const storageConfig = {
      enabled: true,
      dataDir: testDataDir,
      liveDataFile: 'test_live_data.json',
      useTempFile: false
    };
    dataStorage = new DataStorageService(storageConfig, logger);

    // Create HTTPS server with this suite's ephemeral certificate
    const serverOptions = {
      key: fs.readFileSync(testCertificate.keyPath),
      cert: fs.readFileSync(testCertificate.certPath)
    };

    httpsServer = https.createServer(serverOptions, (req, res) => {
      res.writeHead(200);
      res.end('XR Linker Test Server');
    });

    // Initialize Socket.IO service
    const socketConfig = {
      cors: {
        origin: '*',
        methods: ['GET', 'POST']
      },
      pingTimeout: 10000,
      pingInterval: 5000,
      // Keep latency metadata in the frame (part of the file contract) but never
      // write latency logs into the repository's data directory
      latency: {
        enabled: true,
        logToFile: false
      }
    };
    socketService = new SocketService(httpsServer, socketConfig, logger, dataStorage);

    // Start server
    httpsServer.listen(testPort, () => {
      console.log(`Test server running on port ${testPort}`);
      done();
    });
  });

  afterAll((done) => {
    const cleanup = () => {
      if (fs.existsSync(testDataDir)) {
        fs.rmSync(testDataDir, { recursive: true, force: true });
      }
      if (fs.existsSync(testLogDir)) {
        fs.rmSync(testLogDir, { recursive: true, force: true });
      }
      testCertificate.cleanup();
    };

    // Close client connection if exists
    if (client && client.connected) {
      client.disconnect();
    }

    const closeServer = () => {
      if (!httpsServer) {
        cleanup();
        done();
        return;
      }
      // Lingering keep-alive sockets would otherwise hold the listener open
      if (typeof httpsServer.closeAllConnections === 'function') {
        httpsServer.closeAllConnections();
      }
      httpsServer.close(() => {
        cleanup();
        done();
      });
    };

    // Close server
    if (socketService) {
      socketService.close().then(closeServer, closeServer);
    } else {
      closeServer();
    }
  });

  describe('Server Startup', () => {
    test('should start HTTPS server successfully', () => {
      expect(httpsServer).toBeDefined();
      expect(httpsServer.listening).toBe(true);
    });

    test('should initialize Socket.IO service', () => {
      expect(socketService).toBeDefined();
      expect(socketService.io).toBeDefined();
    });

    test('should initialize data storage service', () => {
      expect(dataStorage).toBeDefined();
      expect(fs.existsSync(testDataDir)).toBe(true);
    });
  });

  describe('Socket.IO Connection', () => {
    test('should accept client connections', (done) => {
      client = ioClient(`https://localhost:${testPort}`, {
        rejectUnauthorized: false, // Accept self-signed certificates in tests
        transports: ['websocket']
      });

      client.on('connect', () => {
        expect(client.connected).toBe(true);
        expect(socketService.getConnectedClientCount()).toBe(1);
        done();
      });

      client.on('connect_error', (error) => {
        done(error);
      });
    }, 10000);

    test('should track connected clients', () => {
      const clients = socketService.getConnectedClients();
      expect(clients.length).toBe(1);
      expect(clients[0]).toHaveProperty('id');
      expect(clients[0]).toHaveProperty('connectedAt');
    });
  });

  describe('Data Flow', () => {
    beforeAll((done) => {
      if (!client || !client.connected) {
        client = ioClient(`https://localhost:${testPort}`, {
          rejectUnauthorized: false,
          transports: ['websocket']
        });
        client.on('connect', () => done());
      } else {
        done();
      }
    });

    test('should receive and process controller data', (done) => {
      const testControllerData = {
        timestamp: Date.now(),
        data: [
          {
            id: 'left',
            handedness: 'left',
            buttons: [
              { pressed: false, touched: false, value: 0 },
              { pressed: false, touched: false, value: 0 }
            ],
            axes: [0.0, 0.0],
            pose: {
              position: [0.1, 1.5, -0.5],
              orientation: [0, 0, 0, 1]
            }
          },
          {
            id: 'right',
            handedness: 'right',
            buttons: [
              { pressed: true, touched: true, value: 1.0 },
              { pressed: false, touched: false, value: 0 }
            ],
            axes: [0.5, -0.3],
            pose: {
              position: [-0.1, 1.5, -0.5],
              orientation: [0, 0, 0, 1]
            }
          }
        ]
      };

      // Send controller data
      client.emit('controller-data', testControllerData);

      // Wait for data to be processed
      setTimeout(() => {
        const latestFrame = dataStorage.getLatestFrame();
        expect(latestFrame).not.toBeNull();
        expect(latestFrame.timestamp).toBe(testControllerData.timestamp);
        expect(latestFrame.data.length).toBe(2);
        expect(latestFrame.data[0].handedness).toBe('left');
        expect(latestFrame.data[1].handedness).toBe('right');
        done();
      }, 500);
    }, 10000);

    test('should write data to file', async () => {
      const testData = {
        timestamp: Date.now(),
        data: [
          {
            id: 'test',
            handedness: 'left',
            buttons: [],
            axes: [],
            pose: {
              position: [0, 0, 0],
              orientation: [0, 0, 0, 1]
            }
          }
        ]
      };

      dataStorage.updateLatestFrame(testData);
      const written = await dataStorage.writeLatestData();

      expect(written).toBe(true);

      const dataFilePath = dataStorage.getDataFilePath();
      expect(fs.existsSync(dataFilePath)).toBe(true);

      const fileContent = fs.readFileSync(dataFilePath, 'utf-8');
      const savedData = JSON.parse(fileContent);
      expect(savedData.timestamp).toBe(testData.timestamp);
    });

    test('should persist server latency metadata while keeping the cached frame trimmed', async () => {
      const frame = {
        timestamp: Date.now(),
        data: [{ id: 'left', handedness: 'left', buttons: [], axes: [] }]
      };

      client.emit('controller-data', frame);
      await new Promise((resolve) => setTimeout(resolve, 300));

      const persisted = JSON.parse(fs.readFileSync(dataStorage.getDataFilePath(), 'utf-8'));

      expect(persisted.timestamp).toBe(frame.timestamp);
      expect(persisted.data).toEqual(frame.data);
      expect(persisted._latency).toBeDefined();
      expect(typeof persisted._latency.serverProcessingLatency).toBe('number');

      // The in-memory cache deliberately holds only the required fields
      expect(Object.keys(dataStorage.getLatestFrame()).sort()).toEqual(['data', 'timestamp']);
    });

    test('should handle multiple rapid data updates', (done) => {
      const dataCount = 10;
      let sentCount = 0;

      const sendData = () => {
        if (sentCount < dataCount) {
          const data = {
            timestamp: Date.now() + sentCount,
            data: [{
              id: `frame-${sentCount}`,
              handedness: 'left',
              buttons: [],
              axes: [],
              pose: { position: [0, 0, 0], orientation: [0, 0, 0, 1] }
            }]
          };
          client.emit('controller-data', data);
          sentCount++;
          setTimeout(sendData, 50);
        } else {
          // Check that data was received
          setTimeout(() => {
            const latestFrame = dataStorage.getLatestFrame();
            expect(latestFrame).not.toBeNull();
            expect(latestFrame.data[0].id).toContain('frame-');
            done();
          }, 500);
        }
      };

      sendData();
    }, 15000);
  });

  describe('Client Disconnection', () => {
    test('should handle client disconnection', (done) => {
      const initialCount = socketService.getConnectedClientCount();
      expect(initialCount).toBeGreaterThan(0);

      client.on('disconnect', () => {
        setTimeout(() => {
          const finalCount = socketService.getConnectedClientCount();
          expect(finalCount).toBe(initialCount - 1);
          done();
        }, 100);
      });

      client.disconnect();
    }, 10000);
  });

  describe('Error Handling', () => {
    test('should handle invalid data format gracefully', (done) => {
      const newClient = ioClient(`https://localhost:${testPort}`, {
        rejectUnauthorized: false,
        transports: ['websocket']
      });

      newClient.on('connect', () => {
        // Send invalid data
        newClient.emit('controller-data', {
          invalid: 'data format'
        });

        // Server should not crash
        setTimeout(() => {
          expect(httpsServer.listening).toBe(true);
          newClient.disconnect();
          done();
        }, 500);
      });
    }, 10000);

    test('should handle malformed data', (done) => {
      const newClient = ioClient(`https://localhost:${testPort}`, {
        rejectUnauthorized: false,
        transports: ['websocket']
      });

      newClient.on('connect', () => {
        // Send various malformed data
        newClient.emit('controller-data', null);
        newClient.emit('controller-data', undefined);
        newClient.emit('controller-data', 'string data');
        newClient.emit('controller-data', 12345);

        setTimeout(() => {
          expect(httpsServer.listening).toBe(true);
          newClient.disconnect();
          done();
        }, 500);
      });
    }, 10000);
  });

  describe('Concurrent Connections', () => {
    test('should handle multiple simultaneous clients', (done) => {
      const clients = [];
      const clientCount = 5;
      let connectedCount = 0;

      const checkAllConnected = () => {
        connectedCount++;
        if (connectedCount === clientCount) {
          setTimeout(() => {
            const serverClientCount = socketService.getConnectedClientCount();
            expect(serverClientCount).toBeGreaterThanOrEqual(clientCount);

            // Disconnect all test clients
            clients.forEach(c => c.disconnect());
            done();
          }, 500);
        }
      };

      for (let i = 0; i < clientCount; i++) {
        const client = ioClient(`https://localhost:${testPort}`, {
          rejectUnauthorized: false,
          transports: ['websocket']
        });

        client.on('connect', checkAllConnected);
        clients.push(client);
      }
    }, 15000);
  });

  describe('Broadcasting', () => {
    test('should broadcast messages to all connected clients', (done) => {
      const client1 = ioClient(`https://localhost:${testPort}`, {
        rejectUnauthorized: false,
        transports: ['websocket']
      });

      const client2 = ioClient(`https://localhost:${testPort}`, {
        rejectUnauthorized: false,
        transports: ['websocket']
      });

      let receivedCount = 0;
      const testMessage = { message: 'test broadcast', timestamp: Date.now() };

      const messageHandler = (data) => {
        expect(data).toEqual(testMessage);
        receivedCount++;
        if (receivedCount === 2) {
          client1.disconnect();
          client2.disconnect();
          done();
        }
      };

      let connectedCount = 0;
      const checkBothConnected = () => {
        connectedCount++;
        if (connectedCount === 2) {
          // Both clients connected, now broadcast
          client1.on('test-broadcast', messageHandler);
          client2.on('test-broadcast', messageHandler);

          socketService.broadcast('test-broadcast', testMessage);
        }
      };

      client1.on('connect', checkBothConnected);
      client2.on('connect', checkBothConnected);
    }, 15000);
  });
});
