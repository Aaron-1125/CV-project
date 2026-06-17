#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SPEC_FILE="${SCRIPT_DIR}/stage4_face_effects.spec"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"
APP_PATH="${DIST_DIR}/Stage4FaceEffects.app"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

cd "${REPO_ROOT}"

echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import PyInstaller" >/dev/null

rm -rf "${BUILD_DIR}" "${DIST_DIR}"

"${PYTHON_BIN}" -m PyInstaller \
  --noconfirm \
  --clean \
  --workpath "${BUILD_DIR}" \
  --distpath "${DIST_DIR}" \
  "${SPEC_FILE}"

if [[ ! -d "${APP_PATH}" ]]; then
  echo "ERROR: app bundle was not created: ${APP_PATH}" >&2
  exit 1
fi

EXECUTABLE_PATH="${APP_PATH}/Contents/MacOS/Stage4FaceEffects"
INFO_PLIST="${APP_PATH}/Contents/Info.plist"

if [[ ! -x "${EXECUTABLE_PATH}" ]]; then
  echo "ERROR: executable missing or not executable: ${EXECUTABLE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${INFO_PLIST}" ]]; then
  echo "ERROR: Info.plist missing: ${INFO_PLIST}" >&2
  exit 1
fi

if command -v codesign >/dev/null 2>&1; then
  if codesign --force --deep --sign - "${APP_PATH}"; then
    echo "Ad-hoc codesign completed."
  else
    echo "WARNING: ad-hoc codesign failed; continuing without notarization." >&2
  fi
fi

echo "Built ${APP_PATH}"
ls -lh "${DIST_DIR}"
