# lingua-map
*started August 13, 2019*

lingua-map is a python-based program for automatically generating an html report summarizing the results of a patients' presurgical fMRI. The report contains resulting activation maps and ROI statistics. It was originally designed for epilepsy patients who receive an fMRI to lateralize language function prior to surgery. 

The algorithm is Dockerized with the intent of turning it into a gear for the Flywheel neuroinformatics platform. 

The program takes a BIDs dataset and fMRIPREP results as inputs.It does a general linear model (GLM) analysis of the preprocessed BOLD time series images to create activation maps using FSL commands wrapped in nipype, then calculates laterality statistics in language ROIs informed by the ASFNR white paper: http://www.ajnr.org/content/38/10/E65. From these results, it generates an html report using jinja-based templating. 

It is designed to run on Flywheel, but it can be run locally with [Docker](https://cloud.docker.com/repository/docker/willtack/lingua-map/general).

```
docker pull willtack/presurgical-report:latest
```


For now, run the container like so:

```
docker run -it -v /home/will/Desktop/bids_dataset:/flywheel/v0/bids_dataset \
                         -v /home/will/Desktop/fmriprep:/flywheel/v0/fmriprep
                         --entrypoint /bin/bash willtack/presurgical-report:0.0.8
```

Then run the main script from inside the container, e.g.:
```
python report.py --bidsdir /flywheel/v0/bids_dataset \
                 --fmriprepdir /flywheel/v0/fmriprep \
                 --outputdir /flywheel/v0/ \
                 --aroma
```

Usage:
```
usage: report.py [-h] --bidsdir BIDSDIR --fmriprepdir FMRIPREPDIR --outputdir
                 OUTPUTDIR [--aroma] --tasks TASKS [TASKS ...]

Generate a presurgical report from fmriprep results

optional arguments:
  -h, --help            show this help message and exit
  --bidsdir BIDSDIR     Path to a curated BIDS directory
  --fmriprepdir FMRIPREPDIR Path to fmriprep results directory
  --outputdir OUTPUTDIR Path to output directory
  --tasks TASKS [TASKS ...] Space separated list of tasks
  --aroma               Use ICA-AROMA denoised BOLD images

```
