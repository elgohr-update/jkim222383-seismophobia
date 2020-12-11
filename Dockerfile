## Group 11 Dockerfile
# This Dockerfile will install the versions of R,Python and all packages in both that are required for running analysis.
FROM rocker/tidyverse:4.0.3


ENV RENV_VERSION 0.12.2
RUN R -e "install.packages('renv', version='${RENV_VERSION}')"

# Setup using Renv
WORKDIR /home/seismophobia

# Copy all renv setup files into container for restoring library.
COPY renv.lock .Rprofile ./
COPY renv/settings.dcf renv/activate.R renv/
# Get R packages set to proper versions.
ENV RENV_PATHS_CACHE="/renv/cache"
ENV RENV_CONFIG_USE_CACHE=TRUE  
RUN R -e 'renv::consent(provided=TRUE)'
RUN Rscript -e 'renv::restore()'

# Get Make for re running analysis with Makefile
RUN apt-get update && apt-get install make


# --------------------------------------Conda Setup -----------------------------------------------------------------------------------
# Verbatim from: https://hub.docker.com/r/continuumio/miniconda3/dockerfile

# MiniConda build notes:
#  $ docker build . -t continuumio/miniconda3:latest -t continuumio/miniconda3:4.5.11
#  $ docker run --rm -it continuumio/miniconda3:latest /bin/bash
#  $ docker push continuumio/miniconda3:latest
#  $ docker push continuumio/miniconda3:4.5.11


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Get Conda environment file into setup Python
# Installs packages into base Conda environment
COPY seismophobia_conda_env.yml* /home/seismophobia/
RUN conda env update -n base --file /home/seismophobia/seismophobia_conda_env.yml 

CMD [ "/bin/bash" ]