# Base image
FROM astrocrpublic.azurecr.io/runtime:3.0-6

# Switch to root to install system packages
USER root

# 1) Install system packages
RUN apt-get update \
    && apt-get install -y default-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2) Copy requirements.txt and install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 3) Copy application code and Hydra configuration
COPY dags/ /usr/local/airflow/dags/
COPY include/ /usr/local/airflow/include/

# 4) Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="$JAVA_HOME/bin:$PATH"

# 5) Set Hydra configuration path for containers
ENV HYDRA_CONFIG_PATH=/usr/local/airflow/include/conf
