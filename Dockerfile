FROM ubuntu:16.04
MAINTAINER Will Tackett <william.tackett@pennmedicine.upenn.edu>

# Make directory for flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}

# Set the entrypoint
ENTRYPOINT ["/flywheel/v0/run.sh"]

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    bzip2 \
                    ca-certificates \
                    xvfb \
                    cython3 \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config \
                    jq \
                    zip \
                    git && \
    curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get install -y --no-install-recommends \
                    nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV FSLDIR="/usr/share/fsl"
RUN apt-get update -qq \
  && apt-get install -y -q --no-install-recommends \
         bc \
         dc \
         file \
         libfontconfig1 \
         libfreetype6 \
         libgl1-mesa-dev \
         libglu1-mesa-dev \
         libgomp1 \
         libice6 \
         libxcursor1 \
         libxft2 \
         libxinerama1 \
         libxrandr2 \
         libxrender1 \
         libxt6 \
         wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && echo "Downloading FSL ..." \
  && mkdir -p /usr/share/fsl \
  && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.11-centos6_64.tar.gz \
  | tar -xz -C /usr/share/fsl --strip-components 1 \
  && echo "Installing FSL conda environment ..." \
  && bash /usr/share/fsl/etc/fslconf/fslpython_install.sh -f /usr/share/fsl

ENV PATH="${FSLDIR}/bin:$PATH"

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh && \
    bash Miniconda3-4.5.12-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.5.12-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1

RUN conda install -y mkl=2019.4 mkl-service;  sync &&\
    conda install -y numpy=1.15.4 \
                     scipy=1.3.0 \
                     scikit-learn=0.19.1 \
                     matplotlib=2.2.2 \
                     pandas=0.25.0 \
                     libxml2=2.9.9 \
                     graphviz=2.40.1 \
                     zlib; sync && \
    conda install -c conda-forge traits; sync && \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda build purge-all; sync && \
    conda clean -tipsy && sync

RUN pip install flywheel-sdk pandas
RUN pip install --no-cache fw-heudiconv \
    && pip install --no-cache flywheel-sdk \
    && pip install --no-cache nipype \
    && pip install --no-cache nilearn \
    && pip install --no-cache pybids \
    && pip install --no-cache jinja2 \
    && pip install --no-cache argparse \
    && pip install --no-cache nibabel \
    && pip install --no-cache nistats \
    && pip install --no-cache cairocffi \
    && pip install --no-cache weasyprint

COPY manifest.json ${FLYWHEEL}/manifest.json
COPY run.sh /flywheel/v0/run.sh
COPY . /flywheel/v0/
RUN chmod +x ${FLYWHEEL}/*

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y unzip

WORKDIR /flywheel/v0

RUN conda install matplotlib
RUN conda install scikit-learn
RUN apt-get install -y libcairo2-dev
RUN apt-get install -y pango-1.0