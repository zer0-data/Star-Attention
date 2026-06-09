#!/bin/bash

# Get the absolute directory of this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Define the URL file path using the script directory
URL_FILE="$SCRIPT_DIR/file_download_urls.txt"

# Check if file_download_urls.txt exists in the script directory
if [[ ! -f "$URL_FILE" ]]; then
  echo "Error: file_download_urls.txt not found in $SCRIPT_DIR."
  exit 1
fi

# Loop through each line (URL) in the file
while IFS= read -r url; do
  # Skip empty lines
  if [[ -n "$url" ]]; then
    echo "Downloading: $url"
    wget -P "${SCRIPT_DIR}/task_files" "$url"
  fi
done < "$URL_FILE"