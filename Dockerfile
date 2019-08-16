FROM python:3
MAINTAINER Will Tackett <william.tackett@pennmedicine.upenn.edu>

# Make directory for flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
RUN mkdir -p imgs
RUN mkdir -p masks
RUN mkdir -p bids_dataset
COPY manifest.json ${FLYWHEEL}/manifest.json

# Set the entrypoint
ENTRYPOINT ["/flywheel/v0/run.sh"]

RUN apt-get -y update
RUN apt-get install -y zip

RUN pip install flywheel-sdk \
    && pip install nipype \
    && pip install nilearn \
    && pip install nibabel \
    && pip install matplotlib \
    && pip install bids \
    && pip install numpy \
    && pip install sklearn \
    && pip install fw-heudiconv

# Copy over python scripts
COPY report.py ${FLYWHEEL}/report.py
COPY run.sh ${FLYWHEEL}/run.sh
COPY imgs/ $FLYWHEEL}/imgs/
COPY masks/ ${FLYWHEEL}/masks/
COPY create_archive_fw_heudiconv.py ${FLYWHEEL}/create_archive_fw_heudiconv.py
COPY bids_dataset/ ${FLYWHEEL}/bids_dataset/
RUN chmod +x ${FLYWHEEL}/*

#  ENV preservation for Flywheel Engine
#  RUN env -u HOSTNAME -u PWD | \
#  awk -F = '{ print "export " $1 "=\"" $2 "\"" }' > ${FLYWHEEL}/docker-#env.sh

WORKDIR /flywheel/v0
