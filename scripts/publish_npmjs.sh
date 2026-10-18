#!/usr/bin/env bash
# scripts/publish_npmjs.sh -- publish the unscoped package openclaw-hy-memory to registry.npmjs.org
#
# Sources and the published package both live in plugins/openclaw/.
#
# Usage:
#   bash scripts/publish_npmjs.sh 1.1.8
#
# Authentication (either one works):
#   //registry.npmjs.org/:_authToken=... in ~/.npmrc
#   export NPM_TOKEN=npm_...

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLAW="${ROOT}/plugins/openclaw"
VERSION="${1:-}"

if [[ -z "${VERSION}" ]]; then
  echo "Usage: bash scripts/publish_npmjs.sh <version>"
  echo "Example: bash scripts/publish_npmjs.sh 1.1.8"
  exit 1
fi

echo "→ build plugins/openclaw/"
(cd "${OPENCLAW}" && npm run build)

echo "→ set package version ${VERSION}"
node -e "
const fs = require('fs');
const path = '${OPENCLAW}/package.json';
const pkg = JSON.parse(fs.readFileSync(path, 'utf8'));
pkg.version = '${VERSION}';
fs.writeFileSync(path, JSON.stringify(pkg, null, 2) + '\n');
"

echo "→ publish openclaw-hy-memory@${VERSION} to registry.npmjs.org"
NPM_TOKEN_NPMJS="${NPM_TOKEN:-$(grep '//registry.npmjs.org/:_authToken=' "${HOME}/.npmrc" 2>/dev/null | cut -d= -f2- || true)}"
if [[ -z "${NPM_TOKEN_NPMJS}" ]]; then
  echo "ERROR: NPM_TOKEN or //registry.npmjs.org/:_authToken in ~/.npmrc is required"
  exit 1
fi

(cd "${OPENCLAW}" && npm publish \
  --registry https://registry.npmjs.org/ \
  --//registry.npmjs.org/:_authToken="${NPM_TOKEN_NPMJS}")

echo "✓ Published: openclaw-hy-memory@${VERSION} → https://www.npmjs.com/package/openclaw-hy-memory"
