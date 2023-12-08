# Dockerfile for training dockers with OpenCLIP.
#
# To build:
# docker build -f model_oss/autogluon/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends git
RUN apt-get install -y --no-install-recommends jq
RUN apt-get install -y --no-install-recommends gnupg
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends tesseract-ocr

ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE


# Install libraries.
RUN pip install autogluon==1.0.0

COPY model_oss/autogluon /autogluon
WORKDIR /autogluon

ENTRYPOINT ["python", "train.py"]