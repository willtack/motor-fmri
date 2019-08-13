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
ENTRYPOINT ["/flywheel/v0/run.py"]

RUN apt-get -y update
RUN apt-get install -y zip

RUN pip install flywheel-sdk \
    && pip install nipype \
    && pip install nilearn \
    && pip install nibabel \
    && pip install matplotlib \
    && pip install bids \
    && pip install numpy \
    && pip install sklearn

# Copy over python scripts
COPY report.py /flywheel/v0/report.py
COPY run.py /flywheel/v0/run.py
COPY imgs/ /imgs/
COPY masks/ /masks/
COPY bids_dataset/ /bids_dataset/
RUN chmod +x ${FLYWHEEL}/*

#  ENV preservation for Flywheel Engine
#  RUN env -u HOSTNAME -u PWD | \
#  awk -F = '{ print "export " $1 "=\"" $2 "\"" }' > ${FLYWHEEL}/docker-#env.sh

WORKDIR /flywheel/v0
