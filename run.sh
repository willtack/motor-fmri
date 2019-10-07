#!/bin/bash
#
#
#

FLYWHEEL_BASE=/flywheel/v0
CODE_BASE=${FLYWHEEL_BASE}/code
MANIFEST_FILE=${FLYWHEEL_BASE}/manifest.json
INPUT_DIR=${FLYWHEEL_BASE}/input
mkdir -p ${INPUT_DIR}
OUTPUT_DIR=${FLYWHEEL_BASE}/output
mkdir -p ${OUTPUT_DIR}
CONTAINER='[flywheel/presurgicalreport]'

error_exit()
{
	echo "$@" 1>&2
	exit 1
}

# CREATE A BIDS FORMATTED DIRECTORY
#   Use fw-heudiconv to accomplish this task
/usr/local/miniconda/bin/python3 ${CODE_BASE}/create_archive_fw_heudiconv.py
 if [[ $? != 0 ]]; then
   error_exit "$CONTAINER Problem creating archive! Exiting (1)"
 fi

# VALIDATE INPUT DATA
# Check if the input directory is not empty
if [[ "$(ls -A $INPUT_DIR)" ]] ; then
    echo "$CONTAINER  Starting..."
else
    error_exit "$CONTAINER Input directory is empty: $INPUT_DIR"
fi

# Show the contents of the BIDS directory
BIDS_DIR=${INPUT_DIR}/bids_dataset
ls -R ${BIDS_DIR}

# Get the list of tasks based on what's in the bids dataset
TASK_LIST=$(python ${CODE_BASE}/filter_tasks.py --bidsdir ${BIDS_DIR})

# Position fmriprepdir contents
unzip ${INPUT_DIR}/fmriprepdir/*.zip -d ${INPUT_DIR}
cd ${INPUT_DIR} || error_exit "$CONTAINER Could not enter input directory."
if [ -d ${INPUT_DIR}/fmriprepdir ]; then
  rm -rf ${INPUT_DIR}/fmriprepdir
fi
FMRIPREP_DIR=$(find $(pwd) -maxdepth 2 -type d | grep -E -v bids | grep -E -v fmriprepdir | grep -E fmriprep)
cd ${FLYWHEEL_BASE} || error_exit "$CONTAINER Could not enter /flywheel/v0/"

# Copy event files to bids dataset
cp ${FLYWHEEL_BASE}/events/* ${INPUT_DIR}/bids_dataset/

# Create results directory
SUB_ID=$(find /flywheel/v0/input/bids_dataset -maxdepth 1 -type d | grep sub | cut -d '/' -f 6)
RESULTS_DIR=${FLYWHEEL_BASE}/"${SUB_ID}"_report_results
mkdir -p "${RESULTS_DIR}"

# Copy imgs/ to results directory
cp -r ${FLYWHEEL_BASE}/imgs "${RESULTS_DIR}"/

# Parse configuration
function parse_config {

  CONFIG_FILE=$FLYWHEEL_BASE/config.json
  MANIFEST_FILE=$FLYWHEEL_BASE/manifest.json

  if [[ -f $CONFIG_FILE ]]; then
    echo "$(cat $CONFIG_FILE | jq -r '.config.'"$1")"
  else
    CONFIG_FILE=$MANIFEST_FILE
    echo "$(cat $MANIFEST_FILE | jq -r '.config.'"$1"'.default')"
  fi
}

config_aroma="$(parse_config 'AROMA')"

if [[ $config_aroma == 'false' ]]; then
  aroma_FLAG=''
else
  aroma_FLAG='--aroma'
fi

config_intermediary="$(parse_config 'save_intermediary_files')"
config_thresh_method="$(parse_config 'thresh_method')"
config_fwhm="$(parse_config 'fwhm')"
config_cthresh="$(parse_config 'cluster_size_thresh')"

# Run script
if [[ $config_thresh_method == 'FDR' ]]; then
  /usr/local/miniconda/bin/python3 ${CODE_BASE}/report.py --bidsdir "${BIDS_DIR}" \
                                             --fmriprepdir "${FMRIPREP_DIR}" \
                                             --outputdir "${RESULTS_DIR}"    \
                                             --tasks "${TASK_LIST}"  \
                                             --fwhm "$config_fwhm" \
                                             --cthresh "$config_cthresh" \
                                              ${aroma_FLAG} \
                                              || error_exit "$CONTAINER Main script failed! Check traceback above."
elif [[ $config_thresh_method == 'cluster' ]]; then
  /usr/local/miniconda/bin/python3 ${CODE_BASE}/report_cluster.py --bidsdir "${BIDS_DIR}" \
                                             --fmriprepdir "${FMRIPREP_DIR}" \
                                             --outputdir "${RESULTS_DIR}"    \
                                             --tasks "${TASK_LIST}"  \
                                             --fwhm "$config_fwhm" \
                                             --cthresh "$config_cthresh" \
                                              ${aroma_FLAG} \
                                              || error_exit "$CONTAINER Main script failed! Check traceback above."
else
  error_exit "Could not identify thresholding configuration."
fi

# Position results directory as zip file in /flywheel/v0/output
zip -r "${SUB_ID}"_report_results.zip "${SUB_ID}"_report_results
mv "${SUB_ID}"_report_results.zip ${OUTPUT_DIR}/

# Remove intermediary files (make config?)
if [[ "$config_intermediary" == 'false' ]]; then
  rm -r $(find output -maxdepth 3 -type d | grep modelfit) || echo "No intermediary files to delete."
else
  echo "Saving intermediary files from modelfitting workflow."
fi

# Remove report_results directory from container
rm $(find output -maxdepth 3 -type f | grep cluster_thresholded_z_resample.nii.gz) || echo ""
rm -rf "${RESULTS_DIR}" || echo "No results directory to delete."
rm stat_result.json || echo "stat_result.json not found. No need to remove."
rm tsnr.nii.gz || echo "tsnr.nii.gz not found. No need to remove."

echo "Completed analysis and generated report successfully!"

