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
 * LatencyMonitor - Server-side latency monitoring and logging
 *
 * This module tracks data packets as they pass through the server,
 * recording timestamps and computing latency statistics.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Get __dirname equivalent in ES6 modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class LatencyMonitor {
    constructor(options = {}) {
        this.enabled = options.enabled !== false;
        this.logToFile = options.logToFile !== false;
        this.logFilePath = options.logFilePath || path.join(__dirname, '../../data/latency_log.jsonl');
        this.maxBufferSize = options.maxBufferSize || 1000;

        this.stats = {
            packetsReceived: 0,
            packetsForwarded: 0,
            totalServerLatency: 0,
            minServerLatency: Infinity,
            maxServerLatency: 0,
            latencies: []
        };

        // Buffer for batched file writes
        this.writeBuffer = [];
        this.lastFlush = Date.now();
        this.flushInterval = options.flushInterval || 5000; // Flush every 5 seconds

        // Ensure data directory exists
        if (this.logToFile) {
            const dir = path.dirname(this.logFilePath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
        }

        console.log(`LatencyMonitor initialized (enabled: ${this.enabled}, logToFile: ${this.logToFile})`);
    }

    /**
     * Record when a packet is received from client
     * @param {Object} data - The data packet
     * @returns {Object} - Augmented data with server receive timestamp
     */
    onPacketReceived(data) {
        if (!this.enabled) {
            return data;
        }

        const serverReceiveTime = Date.now();
        const serverReceiveTimeHR = process.hrtime.bigint();

        // Create augmented data with server timestamps
        const augmentedData = {
            ...data,
            _latency: {
                ...(data._latency || {}),
                serverReceiveTime,
                serverReceiveTimeHR: serverReceiveTimeHR.toString(),
                serverReceiveTimeISO: new Date().toISOString()
            }
        };

        // Calculate estimated client-to-server latency if network latency estimate is available
        if (augmentedData._latency.clientTimestamp && augmentedData._latency.estimatedNetworkLatency !== undefined) {
            // Use the estimated network latency from RTT measurements
            augmentedData._latency.estimatedClientToServerLatency = augmentedData._latency.estimatedNetworkLatency;

            // Optionally calculate "time difference" (for debugging clock sync issues)
            // This is NOT the actual latency, just shows clock offset
            augmentedData._latency.clockOffsetDebug = serverReceiveTime - augmentedData._latency.clientTimestamp;
        }

        this.stats.packetsReceived++;

        return augmentedData;
    }

    /**
     * Record when a packet is forwarded/written to file
     * @param {Object} data - The data packet
     * @returns {Object} - Final augmented data with forward timestamp
     */
    onPacketForwarded(data) {
        if (!this.enabled) {
            return data;
        }

        const serverForwardTime = Date.now();
        const serverForwardTimeHR = process.hrtime.bigint();

        // Add forward timestamp
        const finalData = {
            ...data,
            _latency: {
                ...(data._latency || {}),
                serverForwardTime,
                serverForwardTimeHR: serverForwardTimeHR.toString(),
                serverForwardTimeISO: new Date().toISOString()
            }
        };

        // Calculate server processing latency if receive timestamp exists
        if (finalData._latency.serverReceiveTime) {
            const serverLatency = serverForwardTime - finalData._latency.serverReceiveTime;
            finalData._latency.serverProcessingLatency = serverLatency;

            // Update statistics
            this.stats.totalServerLatency += serverLatency;
            this.stats.minServerLatency = Math.min(this.stats.minServerLatency, serverLatency);
            this.stats.maxServerLatency = Math.max(this.stats.maxServerLatency, serverLatency);
            this.stats.latencies.push(serverLatency);

            // Keep only recent latencies to avoid memory issues
            if (this.stats.latencies.length > this.maxBufferSize) {
                this.stats.latencies.shift();
            }
        }

        this.stats.packetsForwarded++;

        // Log to file if enabled
        if (this.logToFile && finalData._latency) {
            this._bufferLogEntry(finalData._latency);
        }

        return finalData;
    }

    /**
     * Handle ping requests for RTT measurement
     * @param {Object} pingData - The ping packet
     * @returns {Object} - Pong response with server timestamp
     */
    handlePing(pingData) {
        const pongData = {
            type: 'pong',
            _latency: {
                ...(pingData._latency || {}),
                serverTimestamp: Date.now(),
                serverTimeISO: new Date().toISOString()
            }
        };

        return pongData;
    }

    /**
     * Buffer log entry for batched writes
     * @private
     */
    _bufferLogEntry(latencyData) {
        this.writeBuffer.push(latencyData);

        // Auto-flush if buffer is full or interval elapsed
        const now = Date.now();
        if (this.writeBuffer.length >= 100 || now - this.lastFlush > this.flushInterval) {
            this.flushLogs();
        }
    }

    /**
     * Flush buffered logs to file
     */
    flushLogs() {
        if (this.writeBuffer.length === 0) {
            return;
        }

        try {
            const logLines = this.writeBuffer.map(entry => JSON.stringify(entry)).join('\n') + '\n';
            fs.appendFileSync(this.logFilePath, logLines);
            this.writeBuffer = [];
            this.lastFlush = Date.now();
        } catch (error) {
            console.error('Error flushing latency logs:', error);
        }
    }

    /**
     * Get current statistics
     * @returns {Object} - Current latency statistics
     */
    getStats() {
        const avgServerLatency = this.stats.packetsForwarded > 0
            ? this.stats.totalServerLatency / this.stats.packetsForwarded
            : 0;

        // Calculate percentiles
        const sortedLatencies = [...this.stats.latencies].sort((a, b) => a - b);
        const p50 = this._percentile(sortedLatencies, 50);
        const p95 = this._percentile(sortedLatencies, 95);
        const p99 = this._percentile(sortedLatencies, 99);

        return {
            packetsReceived: this.stats.packetsReceived,
            packetsForwarded: this.stats.packetsForwarded,
            avgServerLatency: avgServerLatency.toFixed(3),
            minServerLatency: this.stats.minServerLatency === Infinity ? 0 : this.stats.minServerLatency.toFixed(3),
            maxServerLatency: this.stats.maxServerLatency.toFixed(3),
            p50ServerLatency: p50.toFixed(3),
            p95ServerLatency: p95.toFixed(3),
            p99ServerLatency: p99.toFixed(3),
            bufferSize: this.writeBuffer.length
        };
    }

    /**
     * Calculate percentile
     * @private
     */
    _percentile(sortedArray, percentile) {
        if (sortedArray.length === 0) return 0;
        const index = Math.ceil((percentile / 100) * sortedArray.length) - 1;
        return sortedArray[Math.max(0, index)];
    }

    /**
     * Reset statistics
     */
    reset() {
        this.stats = {
            packetsReceived: 0,
            packetsForwarded: 0,
            totalServerLatency: 0,
            minServerLatency: Infinity,
            maxServerLatency: 0,
            latencies: []
        };
        this.writeBuffer = [];
        console.log('LatencyMonitor statistics reset');
    }

    /**
     * Print statistics to console
     */
    logStats() {
        const stats = this.getStats();
        console.log('=== LatencyMonitor Statistics ===');
        console.log(`Packets received: ${stats.packetsReceived}`);
        console.log(`Packets forwarded: ${stats.packetsForwarded}`);
        console.log(`Server latency - Avg: ${stats.avgServerLatency}ms, Min: ${stats.minServerLatency}ms, Max: ${stats.maxServerLatency}ms`);
        console.log(`Server latency - P50: ${stats.p50ServerLatency}ms, P95: ${stats.p95ServerLatency}ms, P99: ${stats.p99ServerLatency}ms`);
        console.log(`Buffer size: ${stats.bufferSize}`);
        console.log('==================================');
    }

    /**
     * Enable or disable monitoring
     */
    setEnabled(enabled) {
        this.enabled = enabled;
        console.log(`LatencyMonitor ${enabled ? 'enabled' : 'disabled'}`);
    }

    /**
     * Cleanup and flush on shutdown
     */
    shutdown() {
        this.flushLogs();
        this.logStats();
        console.log('LatencyMonitor shutdown complete');
    }
}

export default LatencyMonitor;
