## Group 11 Dockerfile
# This Dockerfile will install the versions of R,Python and all packages in both that are required for running analysis.

FROM rocker/tidyverse:4.0.3

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

#----------------------------------Renv Setup -------------------------------------------------------------------------------------

# Setup Constants - we'll use Rstudio's package manager to download binaries to speed things up.
ENV RENV_VERSION 0.12.2

# Manual R package installs. Tidyverse already installed------------------------------------------
RUN install2.r --error \
    --deps TRUE \
    docopt \
    here \
    knitr \ 
    rmarkdown \ 
    ggthemes \
    testthat 

# Setup using Renv
WORKDIR /home/seismophobia

## PREVIOUS ATTEMPT: Copy all renv setup files into container for restoring library.
# COPY renv.lock .Rprofile ./
# COPY renv/settings.dcf renv/activate.R renv/
# # Get R packages set to proper versions.
# ENV RENV_PATHS_CACHE="/renv/cache"
# ENV RENV_CONFIG_USE_CACHE=TRUE  

## NEW ATTEMPT: Initialize using renv inside the container, then copy in renv.lock file for building library
# RUN R -e 'renv::consent(provided=TRUE)'
# RUN R -e "options(renv.config.cache.symlinks = FALSE)"
# RUN R -e 'renv::init(bare=TRUE)'
# COPY renv.lock renv.lock
# RUN Rscript -e 'renv::restore()'
# RUN Rscript -e "renv::isolate()"

# COPY renv.lock renv.lock
# RUN Rscript -e "install.packages('remotes', repos = c(CRAN = Sys.getenv('CRAN_REPO')))"
# RUN Rscript -e "remotes::install_github('rstudio/renv', ref = Sys.getenv('RENV_VERSION'))"
# RUN Rscript -e "renv::restore(repos = c(CRAN = Sys.getenv('CRAN_REPO')))"

#

CMD [ "/bin/bash" ]