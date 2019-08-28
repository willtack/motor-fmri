FROM ubuntu:latest
MAINTAINER Will Tackett <william.tackett@pennmedicine.upenn.edu>

# Pre-cache neurodebian key
COPY neurodebian.gpg /usr/local/etc/neurodebian.gpg

# Make directory for flywheel spec (v0)
ENV FLYWHEEL /flywheel/v0
RUN mkdir -p ${FLYWHEEL}
COPY manifest.json ${FLYWHEEL}/manifest.json

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
                    git && \
    curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get install -y --no-install-recommends \
                    nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Installing Neurodebian packages (FSL, AFNI, git)
RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /usr/local/etc/neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)

# Installing precomputed python packages
ENV CONDA_DIR="/opt/miniconda-latest" \
    PATH="/opt/miniconda-latest/bin:$PATH"
RUN export PATH="/opt/miniconda-latest/bin:$PATH" \
    && echo "Downloading Miniconda installer ..." \
    && conda_installer="/tmp/miniconda.sh" \
    && curl -fsSL --retry 5 -o "$conda_installer" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash "$conda_installer" -b -p /opt/miniconda-latest \
    && rm -f "$conda_installer" \
    && conda update -yq -nbase conda \
    && conda config --system --prepend channels conda-forge \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    && sync && conda clean -tipsy && sync \
    && conda install -y python=3.7.1 \
                         pip=19.1 \
                         mkl=2018.0.3 \
                         mkl-service \
                         numpy=1.15.4 \
                         scipy=1.1.0 \
                         scikit-learn=0.19.1 \
                         matplotlib=2.2.2 \
                         pandas=0.23.4 \
                         libxml2=2.9.8 \
                         libxslt=1.1.32 \
                         graphviz=2.40.1 \
                         traits=4.6.0 \
                         zlib; sync && \
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
    && pip install --no-cache jinja2

COPY run.sh /flywheel/v0/run.sh
COPY . /flywheel/v0/
RUN chmod +x ${FLYWHEEL}/*

# ENV preservation for Flywheel Engine
RUN env -u HOSTNAME -u PWD | \
  awk -F = '{ print "export " $1 "=\"" $2 "\"" }' > ${FLYWHEEL}/docker-env.sh

WORKDIR /flywheel/v0
