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
 * LatencyTracker - Tracks latency by adding sequence numbers and timestamps to data packets
 *
 * This module is used on the PICO VR device to measure end-to-end latency from
 * controller input to the teleop system receiving the data.
 */

class LatencyTracker {
    constructor() {
        this.sequenceNumber = 0;
        this.enabled = true;
        this.stats = {
            packetsSent: 0,
            lastTimestamp: 0,
            startTime: performance.now()
        };
        // RTT tracking for estimating network latency
        this.rttHistory = [];
        this.maxRttHistory = 10;  // Keep last 10 RTT measurements
        this.currentEstimatedNetworkLatency = 0;
    }

    /**
     * Enable or disable latency tracking
     * @param {boolean} enabled - Whether to enable tracking
     */
    setEnabled(enabled) {
        this.enabled = enabled;
        console.log(`LatencyTracker ${enabled ? 'enabled' : 'disabled'}`);
    }

    /**
     * Add latency tracking metadata to a data packet
     * @param {Object} data - The data packet to augment
     * @returns {Object} - The augmented data packet with latency metadata
     */
    addLatencyMetadata(data) {
        if (!this.enabled) {
            return data;
        }

        const now = performance.now();

        // Create a deep copy to avoid modifying the original
        const augmentedData = JSON.parse(JSON.stringify(data));

        // Add latency tracking metadata
        augmentedData._latency = {
            seq: this.sequenceNumber++,
            clientTimestamp: now,
            clientTimeISO: new Date().toISOString(),
            // Also store a high-precision timestamp for better accuracy
            clientTimestampHR: performance.now(),
            // Include estimated network latency based on recent RTT measurements
            estimatedNetworkLatency: this.currentEstimatedNetworkLatency
        };

        this.stats.packetsSent++;
        this.stats.lastTimestamp = now;

        return augmentedData;
    }

    /**
     * Get current tracking statistics
     * @returns {Object} - Current statistics
     */
    getStats() {
        return {
            ...this.stats,
            elapsedTime: performance.now() - this.stats.startTime,
            avgRate: this.stats.packetsSent / ((performance.now() - this.stats.startTime) / 1000)
        };
    }

    /**
     * Reset the tracker
     */
    reset() {
        this.sequenceNumber = 0;
        this.stats = {
            packetsSent: 0,
            lastTimestamp: 0,
            startTime: performance.now()
        };
        console.log('LatencyTracker reset');
    }

    /**
     * Generate a ping packet for RTT measurement
     * @returns {Object} - Ping packet
     */
    generatePingPacket() {
        return {
            type: 'ping',
            _latency: {
                seq: this.sequenceNumber++,
                clientTimestamp: performance.now(),
                clientTimeISO: new Date().toISOString(),
                clientTimestampHR: performance.now()
            }
        };
    }

    /**
     * Process a pong response to calculate RTT
     * @param {Object} pongData - The pong response from server
     * @returns {number} - Round-trip time in milliseconds
     */
    processPongResponse(pongData) {
        if (!pongData._latency) {
            console.warn('Pong packet missing latency metadata');
            return null;
        }

        const now = performance.now();
        const rtt = now - pongData._latency.clientTimestamp;

        // Store RTT in history
        this.rttHistory.push(rtt);
        if (this.rttHistory.length > this.maxRttHistory) {
            this.rttHistory.shift();
        }

        // Calculate average RTT and estimate one-way network latency
        const avgRtt = this.rttHistory.reduce((a, b) => a + b, 0) / this.rttHistory.length;
        this.currentEstimatedNetworkLatency = avgRtt / 2;

        console.log(`RTT: ${rtt.toFixed(2)}ms (seq: ${pongData._latency.seq}), Avg RTT: ${avgRtt.toFixed(2)}ms, Est. Network Latency: ${this.currentEstimatedNetworkLatency.toFixed(2)}ms`);

        return rtt;
    }

    /**
     * Get the current estimated network latency (one-way)
     * Based on average of recent RTT measurements
     * @returns {number} - Estimated one-way network latency in milliseconds
     */
    getEstimatedNetworkLatency() {
        return this.currentEstimatedNetworkLatency;
    }

    /**
     * Log current statistics to console
     */
    logStats() {
        const stats = this.getStats();
        console.log('=== LatencyTracker Statistics ===');
        console.log(`Packets sent: ${stats.packetsSent}`);
        console.log(`Elapsed time: ${(stats.elapsedTime / 1000).toFixed(2)}s`);
        console.log(`Average rate: ${stats.avgRate.toFixed(2)} packets/s`);
        console.log(`Last timestamp: ${stats.lastTimestamp.toFixed(2)}ms`);
        console.log('================================');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LatencyTracker;
}
