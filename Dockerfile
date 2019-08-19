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
RUN apt-get install -y --no-install-recommends \
                    fsl-core=5.0.9-5~nd16.04+1 \
                    zip

RUN pip install flywheel-sdk \
    && pip install nipype \
    && pip install nilearn \
    && pip install nibabel \
    && pip install matplotlib \
    && pip install bids \
    && pip install numpy \
    && pip install sklearn \
    && pip install fw-heudiconv \
    && pip install jinja2

# Copy over python scripts
COPY report.py ${FLYWHEEL}/report.py
COPY run.sh ${FLYWHEEL}/run.sh
COPY imgs/ $FLYWHEEL}/imgs/
COPY masks/ ${FLYWHEEL}/masks/
COPY bids_dataset/ ${FLYWHEEL}/bids_dataset/
COPY outputs/ ${FLYWHEEL}/outputs/
COPY templates/ ${FLYWHEEL}/templates/
RUN chmod +x ${FLYWHEEL}/*

#  ENV preservation for Flywheel Engine
#  RUN env -u HOSTNAME -u PWD | \
#  awk -F = '{ print "export " $1 "=\"" $2 "\"" }' > ${FLYWHEEL}/docker-#env.sh

WORKDIR /flywheel/v0
