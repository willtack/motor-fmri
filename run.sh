#!/bin/bash

FLYWHEEL_BASE=/flywheel/v0
CONTAINER='[flywheel/presurgical-report]'

python ${FLYWHEEL_BASE}/create_archive_fw_heudiconv.py
if [[ $? != 0 ]]; then
  echo "$CONTAINER  Problem creating archive! Exiting (1)"
  exit 1
fi

python report.py --input_path /v0/flywheel/inputs --output_path /v0/flywheel/outputs