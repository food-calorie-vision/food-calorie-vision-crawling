#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data/label-studio"
IMAGE="${LABEL_STUDIO_IMAGE:-heartexlabs/label-studio:latest}"
PORT="${LABEL_STUDIO_PORT:-8080}"
CONTAINER_NAME="${LABEL_STUDIO_CONTAINER:-label-studio-food-calorie}"
LOCAL_ALIAS="${LABEL_STUDIO_LOCAL_ALIAS:-workspace}"
LOCAL_FILES_ROOT="${DATA_ROOT}/local_storage"

mkdir -p "${DATA_ROOT}"
chmod 777 "${DATA_ROOT}"
mkdir -p "${LOCAL_FILES_ROOT}/${LOCAL_ALIAS}"

echo "[Label Studio] Mounting ${PROJECT_ROOT} -> /workspace"
echo "[Label Studio] Persistent data dir: ${DATA_ROOT}"
echo "[Label Studio] Image: ${IMAGE}"
echo "[Label Studio] Port: ${PORT}"
echo "[Label Studio] Local-files alias: ${LOCAL_ALIAS}"

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8080" \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/local_storage \
  -e LABEL_STUDIO_HOST=0.0.0.0 \
  -e LABEL_STUDIO_STATIC_ROOT=/static \
  -v "${PROJECT_ROOT}:/workspace" \
  -v "${PROJECT_ROOT}:/label-studio/data/local_storage/${LOCAL_ALIAS}" \
  -v "${DATA_ROOT}:/label-studio/data" \
  "${IMAGE}" "$@"
