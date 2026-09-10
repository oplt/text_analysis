FROM rocker/r-ver:4.4.0
WORKDIR /opt/r_engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*
COPY r_engine/renv.lock r_engine/.Rprofile ./
COPY r_engine/R ./R
COPY r_engine/run_analysis.R ./
ENV LIBARROW_BINARY=true
RUN R -q -e 'install.packages("renv", repos="https://cloud.r-project.org"); renv::restore(prompt = FALSE)'

# Build from repository root: docker build -f r_engine/Dockerfile -t text-analysis-r-runtime .
# Use this pinned image as the R layer in the dedicated Python 3.12 Celery worker;
# keeping the backend dependency install in that image preserves its uv lock.
ENV RESEARCH_R_ENABLED=true \
    RESEARCH_RSCRIPT_PATH=Rscript \
    RESEARCH_R_ENGINE_ENTRYPOINT=/opt/r_engine/run_analysis.R
CMD ["Rscript", "--version"]
