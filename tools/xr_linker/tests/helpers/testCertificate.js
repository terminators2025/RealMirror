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
 * Ephemeral TLS material for the integration suite.
 *
 * The repository no longer ships keys, so the integration tests mint their own
 * short-lived self-signed certificate for `localhost` in the OS temp directory
 * and delete it afterwards. Nothing is written into `cert/`, and a developer's
 * real certificate is never read or overwritten.
 *
 * Generation needs the `openssl` CLI. When it is unavailable the caller is
 * expected to skip the HTTPS suite rather than fail it.
 */
import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

const OPENSSL_CONFIG = [
  'req',
  '-x509',
  '-newkey',
  'rsa:2048',
  '-nodes',
  '-days',
  '1',
  '-subj',
  '/CN=localhost'
];

const SUBJECT_ALT_NAME = 'subjectAltName=DNS:localhost,IP:127.0.0.1';

/**
 * Creates a self-signed certificate valid for one day.
 *
 * The subject alternative name is optional: older OpenSSL releases lack
 * `-addext`, and the test client disables certificate verification anyway.
 *
 * @param {string} dir - Directory that receives `key.pem` and `cert.pem`.
 * @returns {{keyPath: string, certPath: string, withSubjectAltName: boolean}}
 */
function generate(dir) {
  const keyPath = path.join(dir, 'key.pem');
  const certPath = path.join(dir, 'cert.pem');
  const baseArgs = [...OPENSSL_CONFIG, '-keyout', keyPath, '-out', certPath];

  try {
    execFileSync('openssl', [...baseArgs, '-addext', SUBJECT_ALT_NAME], { stdio: 'pipe' });
    return { keyPath, certPath, withSubjectAltName: true };
  } catch (error) {
    execFileSync('openssl', baseArgs, { stdio: 'pipe' });
    return { keyPath, certPath, withSubjectAltName: false };
  }
}

/**
 * Prepares throwaway TLS material for the integration suite.
 *
 * @returns {{ok: true, dir: string, keyPath: string, certPath: string, cleanup: function(): void}
 *   | {ok: false, reason: string, cleanup: function(): void}}
 */
export function createTestCertificate() {
  let dir;

  try {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'xr-linker-test-tls-'));
    const { keyPath, certPath } = generate(dir);
    return {
      ok: true,
      dir,
      keyPath,
      certPath,
      cleanup: () => fs.rmSync(dir, { recursive: true, force: true })
    };
  } catch (error) {
    if (dir) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
    const detail = error && error.code === 'ENOENT'
      ? 'openssl was not found on PATH'
      : (error && error.message ? error.message.split('\n')[0] : String(error));
    return {
      ok: false,
      reason: detail,
      cleanup: () => {}
    };
  }
}
