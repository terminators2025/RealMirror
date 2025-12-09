# XR Linker - Production Build

This is the compiled production version of XR Linker with protected source code.

## Installation

```bash
npm install --production
```

## Usage

```bash
npm start
```

Or directly:

```bash
node loader.js
```

## Configuration

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

## Requirements

- Node.js >= 14.x(node version == v22.19.0)
- HTTPS certificates in `cert/` directory
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 22.19.0
nvm use 22.19.0
```

## Note

This version contains compiled bytecode. Source code is not included.
