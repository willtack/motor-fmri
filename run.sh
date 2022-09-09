#!/bin/bash
#
#
#

FLYWHEEL_BASE=/flywheel/v0
CONTAINER='[flywheel/presurgicalreport]'
CODE_BASE=${FLYWHEEL_BASE}/src
MANIFEST_FILE=${FLYWHEEL_BASE}/manifest.json
INPUT_DIR=${FLYWHEEL_BASE}/input
OUTPUT_DIR=${FLYWHEEL_BASE}/output
mkdir -p ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}

function error_exit()
{
	echo "$@" 1>&2
	exit 1
}
function parse_config {
  # Parse the config file
  CONFIG_FILE=$FLYWHEEL_BASE/config.json
  MANIFEST_FILE=$FLYWHEEL_BASE/manifest.json

  if [[ -f $CONFIG_FILE ]]; then
    echo "$(cat $CONFIG_FILE | jq -r '.config.'"$1")"
  else
    CONFIG_FILE=$MANIFEST_FILE
    echo "$(cat $MANIFEST_FILE | jq -r '.config.'"$1"'.default')"
  fi
}
function cleanup {
  # Remove report_results directory and other from container
	rm ${FLYWHEEL_BASE}/stat_result.json || echo " stat_result.json not found. No need to remove."
	rm ${FLYWHEEL_BASE}/stdev.nii.gz || echo " stdev.nii.gz not found. No need to remove."
  rm ${FLYWHEEL_BASE}/tsnr.nii.gz || echo "mean_tsnr.nii.gz not found. No need to remove."
	rm ${FLYWHEEL_BASE}/*_input_functional_masked* || echo "Masked images not found. No need to remove."
	rm ${FLYWHEEL_BASE}/*.csv || echo "CSVs not found. No need to remove."
}

# Transfer Docker ENVs to Flywheel Environment
source ${FLYWHEEL_BASE}/docker-env.sh

# Download a BIDs directory using fw-heudiconv
BIDS_DIR=${INPUT_DIR}/bids_dataset
if [[ ! -d ${BIDS_DIR} ]]; then
  /usr/local/miniconda/bin/python3 ${CODE_BASE}/create_archive_fw_heudiconv.py ||  error_exit "$CONTAINER Problem creating archive! Exiting (1)"
fi

# Move events files to bids_dataset top-level if necessary
cp ${FLYWHEEL_BASE}/*.tsv ${INPUT_DIR}/bids_dataset/ || echo "no events files moved"

ls -R ${BIDS_DIR}
echo "$CONTAINER  Starting..."

# Get the list of tasks based on what's in the bids dataset
#TASK_LIST=$(/usr/local/miniconda/bin/python3 ${CODE_BASE}/filter_tasks.py --bidsdir ${BIDS_DIR})
TASK_LIST='motor'
# Position fmriprepdir contents
# See if it's already been extracted (for testing purposes). If not, unzip and look in the appropriate location
FMRIPREP_DIR=$(find $(pwd) -maxdepth 3 -type d | grep -E -v bids | grep -E -v fmriprepdir | grep -E fmriprep)
if [[ ! -d ${FMRIPREP_DIR} ]]; then
  unzip ${INPUT_DIR}/fmriprepdir/*.zip -d ${INPUT_DIR}
  FMRIPREP_DIR=$(find $(pwd) -maxdepth 3 -type d | grep -E -v bids | grep -E -v fmriprepdir | grep -E fmriprep)
fi

# apply fmriprep MNI --> T1w transform to bring the masks into subject space
#anatDir=$(find "${FMRIPREP_DIR}" -type d | grep anat) || error_exit "could not find anat directory in fmriprep folder"
#transform=${anatDir}/sub-*_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5
#mkdir ${FLYWHEEL_BASE}/masks/native
#if [[ -z $transform ]]; then #check that the transform exists
#  for mask in "${FLYWHEEL_BASE}"/masks/*.nii.gz; do
#   maskBaseName=$(basename "$mask" .nii.gz)
#   antsApplyTransforms -d 3 -e 0 -i "$mask" -r "${anatDir}"/sub-P089_desc-preproc_T1w.nii.gz \
#                       -o ${FLYWHEEL_BASE}/masks/native/"${maskBaseName}"_native.nii.gz -t "$transform"
#
#  # threshold and binarize
#   fslmaths ${FLYWHEEL_BASE}/masks/native/"${maskBaseName}"_native.nii.gz -thr 0.5 -bin ${FLYWHEEL_BASE}/masks/native/"${maskBaseName}"_native.nii.gz
#  done
#fi

# Copy event files to bids dataset
#cp ${FLYWHEEL_BASE}/events/* ${INPUT_DIR}/bids_dataset/

# Create results directory
SUB_ID=$(find /flywheel/v0/input/bids_dataset -maxdepth 1 -type d | grep sub | cut -d '/' -f 6)
RESULTS_DIR=${FLYWHEEL_BASE}/"${SUB_ID}"_report_results
mkdir -p "${RESULTS_DIR}"

# Copy imgs/ to results directory
cp -r ${FLYWHEEL_BASE}/imgs "${RESULTS_DIR}"/

# Arg parsing
config_aroma="$(parse_config 'AROMA')"
config_fwhm="$(parse_config 'fwhm')"
config_cthresh="$(parse_config 'cluster_size_thresh')"
config_alpha="$(parse_config 'alpha')"
if [[ $config_aroma == 'false' ]]; then aroma_FLAG=''; else aroma_FLAG='--aroma'; fi

# Run script
/usr/local/miniconda/bin/python3 ${CODE_BASE}/report.py --bidsdir "${BIDS_DIR}" \
                                           --fmriprepdir "${FMRIPREP_DIR}" \
                                           --outputdir "${RESULTS_DIR}"    \
                                           --fwhm "${config_fwhm}" \
                                           --cthresh "${config_cthresh}" \
                                           --alpha "${config_alpha}" \
                                            ${aroma_FLAG} \
                                            || error_exit "$CONTAINER Main script failed! Check traceback above."


# Remove unnecessary files
cleanup

# Copy html report to its own directory
HTML_DIR=${OUTPUT_DIR}/"${SUB_ID}"_report_html
mkdir -p "${HTML_DIR}"
cp "${RESULTS_DIR}"/"${SUB_ID}"_report.html "${HTML_DIR}"
mv "${HTML_DIR}"/"${SUB_ID}"_report.html "${HTML_DIR}"/index.html
for task in ${TASK_LIST}; do
  mkdir "${HTML_DIR}"/"${task}"
  cp -r $(find "${RESULTS_DIR}"/"${task}" -type d  | grep -E figs) "${HTML_DIR}"/"${task}"
done
cp -r ${FLYWHEEL_BASE}/imgs "${HTML_DIR}"
cd ${OUTPUT_DIR}/"${SUB_ID}"_report_html && \
  zip -r ${OUTPUT_DIR}/"${SUB_ID}"_report.html.zip ./* && \
  cd ${FLYWHEEL_BASE} || echo "Unable to zip html directory."
rm -rf "${HTML_DIR}"

# Concatenate csv files and copy to output directory for easy download
out_csv_file="${SUB_ID}_csv.csv"
for filename in $(find "${RESULTS_DIR}" -type f | grep data | grep .csv | grep -Ev scenemem); do
  if [ "$filename" != "$out_csv_file" ] ;
   then
      cat "$filename" >> "$out_csv_file"
  fi
done

sed -n '1~2!p' "$out_csv_file" > "${SUB_ID}_language_data.csv" # delete the headers (every other row)
cp "${SUB_ID}_language_data.csv" ${OUTPUT_DIR}/
cp "${RESULTS_DIR}"/scenemem/stats/scenemem_data.csv ${OUTPUT_DIR}/"${SUB_ID}_scenemem_data.csv"

# Position results directory as zip file in /flywheel/v0/output
zip -r "${SUB_ID}"_report_results.zip "${SUB_ID}"_report_results/*
mv "${SUB_ID}"_report_results.zip ${OUTPUT_DIR}/


rm -rf "${RESULTS_DIR}" || echo "No results directory to delete."

echo "Completed analysis and generated report successfully!"
