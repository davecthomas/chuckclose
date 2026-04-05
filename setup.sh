#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_TEMPLATE_FILE="${PROJECT_ROOT}/env_template.txt"
ENV_FILE="${PROJECT_ROOT}/.env"

log_info() {
  printf 'INFO: %s\n' "$1"
}

log_warn() {
  printf 'WARN: %s\n' "$1"
}

log_error() {
  printf 'ERROR: %s\n' "$1" >&2
}

ensure_poetry() {
  if command -v poetry >/dev/null 2>&1; then
    return
  fi
  log_error "Poetry is required but was not found on PATH."
  log_error "Install Poetry from https://python-poetry.org/docs/#installation and rerun ./setup.sh."
  exit 1
}

seed_env_file_if_missing() {
  if [[ -f "${ENV_FILE}" ]]; then
    return
  fi
  if [[ ! -f "${ENV_TEMPLATE_FILE}" ]]; then
    log_error "env_template.txt was not found at ${ENV_TEMPLATE_FILE}."
    exit 1
  fi

  cp "${ENV_TEMPLATE_FILE}" "${ENV_FILE}"
  log_warn "Created .env from env_template.txt. Fill in GOOGLE_GEMINI_API_KEY before using storyboard video mode."
}

get_setting_value() {
  local setting_name="$1"
  local setting_value="${!setting_name-}"
  if [[ -n "${setting_value}" ]]; then
    printf '%s' "${setting_value}"
    return
  fi
  if [[ ! -f "${ENV_FILE}" ]]; then
    return
  fi

  local matched_line
  matched_line="$(grep -E "^${setting_name}=" "${ENV_FILE}" | tail -n 1 || true)"
  if [[ -z "${matched_line}" ]]; then
    return
  fi
  printf '%s' "${matched_line#*=}"
}

ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return
  fi

  log_warn "ffmpeg was not found on PATH. Attempting to install it."
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
    return
  fi
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg
    return
  fi

  log_error "Unable to install ffmpeg automatically on this system."
  log_error "Install ffmpeg manually and rerun ./setup.sh."
  exit 1
}

install_python_dependencies() {
  log_info "Installing Python dependencies with Poetry."
  poetry install
}

validate_python_runtime() {
  log_info "Validating optional video frame extraction dependencies."
  poetry run python - <<'PY'
import importlib

required_modules = [
    "ai_api_unified",
    "imageio",
    "imageio_ffmpeg",
]

for module_name in required_modules:
    importlib.import_module(module_name)
PY
}

validate_google_video_settings() {
  local video_engine
  local auth_method
  local api_key

  video_engine="$(get_setting_value "VIDEO_ENGINE")"
  auth_method="$(get_setting_value "GOOGLE_AUTH_METHOD")"
  api_key="$(get_setting_value "GOOGLE_GEMINI_API_KEY")"

  if [[ -z "${video_engine}" ]]; then
    log_error "VIDEO_ENGINE is not configured. Set VIDEO_ENGINE=google-gemini in .env or your shell."
    exit 1
  fi
  if [[ "${video_engine}" != "google-gemini" ]]; then
    log_error "This repository is currently configured for Google video generation only. VIDEO_ENGINE must be google-gemini."
    exit 1
  fi
  if [[ -z "${auth_method}" ]]; then
    log_error "GOOGLE_AUTH_METHOD is not configured. Set GOOGLE_AUTH_METHOD=api_key in .env or your shell."
    exit 1
  fi
  if [[ "${auth_method}" != "api_key" ]]; then
    log_error "GOOGLE_AUTH_METHOD must be api_key for the current local workflow."
    exit 1
  fi
  if [[ -z "${api_key}" || "${api_key}" == \[* ]]; then
    log_error "GOOGLE_GEMINI_API_KEY is missing or still contains the placeholder template value."
    log_error "Update ${ENV_FILE} or export GOOGLE_GEMINI_API_KEY in your shell, then rerun ./setup.sh."
    exit 1
  fi
}

main() {
  log_info "Bootstrapping the mosaic development environment."
  ensure_poetry
  seed_env_file_if_missing
  ensure_ffmpeg
  install_python_dependencies
  validate_python_runtime
  validate_google_video_settings
  log_info "Environment setup is complete for Google storyboard video generation."
}

main "$@"
