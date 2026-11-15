#!/usr/bin/env bash
# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#
# Generate a local self-signed certificate for the XR Linker HTTPS server.
#
# WebXR requires a secure context, so the headset must reach this server over
# HTTPS. A self-signed certificate is enough for local teleoperation: the
# browser shows a one-time warning that has to be accepted on the headset.
#
# Usage:
#   bash scripts/gen_cert.sh                 # auto-detect the LAN IP for the SAN
#   bash scripts/gen_cert.sh 192.168.1.105   # pin the IP used for the SAN
#   bash scripts/gen_cert.sh --force         # overwrite an existing certificate
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/cert"
KEY_PATH="${CERT_DIR}/key.pem"
CERT_PATH="${CERT_DIR}/cert.pem"
DAYS=3650

FORCE=false
HOST_IP=""
for arg in "$@"; do
    case "${arg}" in
        --force|-f) FORCE=true ;;
        -h|--help) sed -n '17,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) HOST_IP="${arg}" ;;
    esac
done

if ! command -v openssl >/dev/null 2>&1; then
    echo "[ERROR] openssl not found. Install it with: sudo apt-get install -y openssl" >&2
    exit 1
fi

if [[ -f "${KEY_PATH}" && -f "${CERT_PATH}" && "${FORCE}" != "true" ]]; then
    echo "[INFO] Certificate already exists: ${CERT_PATH}"
    echo "       Re-run with --force to regenerate it."
    exit 0
fi

if [[ -z "${HOST_IP}" ]]; then
    HOST_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')" || true
    [[ -z "${HOST_IP}" ]] && HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')" || true
    [[ -z "${HOST_IP}" ]] && HOST_IP="127.0.0.1"
fi

mkdir -p "${CERT_DIR}"

echo "[INFO] Generating self-signed certificate"
echo "       subject : CN=realmirror-xr-linker"
echo "       SAN     : DNS:localhost, IP:127.0.0.1, IP:${HOST_IP}"
echo "       validity: ${DAYS} days"

openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "${KEY_PATH}" \
    -out "${CERT_PATH}" \
    -days "${DAYS}" \
    -subj "/C=CN/O=RealMirror/CN=realmirror-xr-linker" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${HOST_IP}" \
    >/dev/null 2>&1

chmod 600 "${KEY_PATH}"
chmod 644 "${CERT_PATH}"

echo "[OK] key : ${KEY_PATH}"
echo "[OK] cert: ${CERT_PATH}"
echo ""
echo "Next steps:"
echo "  1. Open https://${HOST_IP}:8080 on the headset and accept the warning once."
echo "  2. If the workstation IP changes, regenerate with:"
echo "     bash scripts/gen_cert.sh --force"
