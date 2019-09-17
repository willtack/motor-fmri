#!/bin/bash
#
#
#

FLYWHEEL_BASE=/flywheel/v0
MANIFEST_FILE=${FLYWHEEL_BASE}/manifest.json
INPUT_DIR=${FLYWHEEL_BASE}/input
mkdir -p ${INPUT_DIR}
OUTPUT_DIR=${FLYWHEEL_BASE}/output
mkdir -p ${OUTPUT_DIR}
RESULTS_DIR=${FLYWHEEL_BASE}/report_results
mkdir -p ${RESULTS_DIR}
CONTAINER='[flywheel/presurgicalreport]'

# CREATE A BIDS FORMATTED DIRECTORY
#   Use fw-heudiconv to accomplish this task
/usr/local/miniconda/bin/python3 ${FLYWHEEL_BASE}/create_archive_fw_heudiconv.py
 if [[ $? != 0 ]]; then
   echo "$CONTAINER  Problem creating archive! Exiting (1)"
   exit 1
 fi

# VALIDATE INPUT DATA
# Check if the input directory is not empty
if [[ "$(ls -A $INPUT_DIR)" ]] ; then
    echo "$CONTAINER  Starting..."
else
    echo "Input directory is empty: $INPUT_DIR"
    exit 1
fi

# Show the contents of the BIDS directory
BIDS_DIR=${INPUT_DIR}/bids_dataset
ls -R ${BIDS_DIR}

# Position fmriprepdir contents
unzip ${INPUT_DIR}/fmriprepdir/*.zip -d ${INPUT_DIR}
cd ${INPUT_DIR} || exit
if [ ! -d ${INPUT_DIR}/fmriprepdir ]; then
rm -rf ${INPUT_DIR}/fmriprepdir
fi
FMRIPREP_DIR=$(find $(pwd) -maxdepth 2 -type d | grep -E -v bids | grep -E -v fmriprepdir | grep -E fmriprep)
cd ${FLYWHEEL_BASE} || exit

# Copy event files
cp ${FLYWHEEL_BASE}/events/* ${INPUT_DIR}/bids_dataset/

# Run script
/usr/local/miniconda/bin/python3 report.py "${BIDS_DIR}" "${FMRIPREP_DIR}" "${RESULTS_DIR}"

# Copy report to output directory
cp ${RESULTS_DIR}/report.html ${OUTPUT_DIR}/

# Position results directory as zip file in /flywheel/v0/output
zip -r report_results.zip report_results
mv report_results.zip ${OUTPUT_DIR}/

# Remove report_results directory from container
rm -rf ${RESULTS_DIR}
