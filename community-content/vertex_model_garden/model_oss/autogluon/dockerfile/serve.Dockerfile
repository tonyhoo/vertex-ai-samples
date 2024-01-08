# Dockerfile for serving dockers with AutoGluon.
#
# To build:
# docker build -f model_oss/autogluon/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim \
        libgomp1  # AutoGluon might require libgomp for some dependencies

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install AutoGluon and other dependencies.
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install autogluon.tabular==1.0.0
RUN python3 -m pip install flask waitress

# Copy your scripts into the container.
COPY model_oss/autogluon /autogluon
WORKDIR /autogluon
# If you have additional utility files or a directory, copy them as well.
# COPY util /util

# Expose the port the app runs on
EXPOSE 8501

# Set the working directory to a specific path for consistency
WORKDIR /autogluon

# Change to a non-root user for security purposes
RUN useradd -m autogluonuser
USER autogluonuser

# Run Flask application
CMD ["python", "serve.py"]