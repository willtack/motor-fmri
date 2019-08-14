#!/bin/bash

bash fw-heudiconv-export --project my_project --path /v0/flywheel/inputs

python report.py --input_path /v0/flywheel/inputs --output_path /v0/flywheel/outputs