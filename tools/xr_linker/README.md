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

- Node.js >= 14.x
- HTTPS certificates in `cert/` directory

## Note

This version contains compiled bytecode. Source code is not included.
