#!/usr/bin/env bash
set -euo pipefail

data_dir="${1:-./data}"
archive_name="speech_commands_v0.02.tar.gz"
url="${GSPEECH_V2_URL:-https://storage.googleapis.com/download.tensorflow.org/data/${archive_name}}"
archive_path="${data_dir}/${archive_name}"

mkdir -p "${data_dir}"

if command -v curl >/dev/null 2>&1; then
    curl --location --fail --continue-at - "${url}" --output "${archive_path}"
elif command -v wget >/dev/null 2>&1; then
    wget --continue "${url}" --output-document "${archive_path}"
else
    echo "Neither curl nor wget is available." >&2
    exit 1
fi

tar --extract --gzip --file "${archive_path}" --directory "${data_dir}"

echo "Google Speech Commands V2 extracted to ${data_dir}"
