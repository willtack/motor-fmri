FROM python:3
MAINTAINER Will Tackett <william.tackett@pennmedicine.upenn.edu>

# Make directory for flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
COPY manifest.json ${FLYWHEEL}/manifest.json

# Set the entrypoint
ENTRYPOINT [/flywheel/v0/run.sh]

RUN apt-get -y update
RUN apt-get install -y zip
RUN pip install flywheel-sdk pandas
COPY . /src
RUN cd /src \
    && pip install . \
    && pip install --no-cache fw-heudiconv \
    && pip install --no-cache flywheel-sdk \
    && pip install --no-cache nipype \
    && pip install --no-cache scikit-learn \
    && pip install --no-cache scikit-image \
    && pip install --no-cache numpy \
    && pip install --no-cache pybids \
    && pip install --no-cache jinja2 \
    && rm -rf /src \
    && apt-get install -y --no-install-recommends zip \
                        fsl-core=5.0.9-5~nd16.04+1

COPY run.sh /flywheel/v0/run.sh
COPY . /flywheel/v0/
RUN chmod +x ${FLYWHEEL}/*

# ENV preservation for Flywheel Engine
RUN env -u HOSTNAME -u PWD | \
  awk -F = '{ print "export " $1 "=\"" $2 "\"" }' > ${FLYWHEEL}/docker-env.sh

WORKDIR /flywheel/v0
