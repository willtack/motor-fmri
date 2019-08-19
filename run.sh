#!/bin/bash

FLYWHEEL_BASE=/flywheel/v0
CONTAINER='[flywheel/presurgical-report]'

#bash fw-heudiconv-export --project my_project --path flywheel/v0/

python report.py /flywheel/v0/bids_dataset /flywheel/v0/bids_dataset/derivatives/fmriprep/ /flywheel/v0/outputs