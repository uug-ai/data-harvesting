# Use the official Python 3.10 slim-bookworm as base image
FROM python:3.10-slim-bookworm

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages and clean up (clean-up added, last line)
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-pip git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /usr/src/ultralytics

# Clone the repository
RUN git clone https://github.com/ultralytics/ultralytics -b main /usr/src/ultralytics

# Add yolov8n.pt model
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt /usr/src/ultralytics/

# Add helmet model
COPY ./models/helmet_dectector_1k_16b_150e.pt /usr/src/ultralytics/

# Copy requirements.txt including the needed python packages
COPY requirements.txt /ml/requirements.txt

# Upgrade pip and install pip packages
RUN python3 -m pip install --upgrade pip wheel && \
    python3 -m pip install --no-cache-dir -r /ml/requirements.txt

# Create necessary directories
RUN mkdir -p /ml/data/input /ml/data/output

# Set working directory
WORKDIR /ml

# Copy application code
COPY . .

# Environment variables
# Feature parameters
ENV PROJECT_NAME=""

# Dataset parameters
ENV DATASET_FORMAT="base"
ENV DATASET_VERSION="1"
ENV DATASET_UPLOAD="True"

# Forwarding
ENV FORWARDING_MEDIA="True"
ENV REMOVE_AFTER_PROCESSED="True"

# Queue parameters
ENV QUEUE_NAME ""
ENV TARGET_QUEUE_NAME ""
ENV QUEUE_EXCHANGE ""
ENV QUEUE_HOST ""
ENV QUEUE_USERNAME ""
ENV QUEUE_PASSWORD ""

# Kerberos Vault parameters
ENV STORAGE_URI ""
ENV STORAGE_ACCESS_KEY ""
ENV STORAGE_SECRET_KEY ""

#Integration parameters
ENV INTEGRATION_NAME=""

# Roboflow parameters
ENV RBF_API_KEY: ""
ENV RBF_WORKSPACE: ""
ENV RBF_PROJECT: ""

#S3 parameters
ENV S3_ENDPOINT=""
ENV S3_ACCESS_KEY=""
ENV S3_SECRET_KEY=""
ENV S3_BUCKET=""

ENV CREATE_BBOX_FRAME "False"
ENV SAVE_BBOX_FRAME "False"
ENV BBOX_FRAME_SAVEPATH "/ml/data/output/output_bbox_frame.jpg"

ENV CREATE_RETURN_JSON "False"
ENV SAVE_RETURN_JSON "False"
ENV RETURN_JSON_SAVEPATH "/ml/data/output/output_json.json"

ENV TIME_VERBOSE "True"
ENV LOGGING "True"

# Classification parameters
ENV CLASSIFICATION_FPS ""
ENV CLASSIFICATION_THRESHOLD ""
ENV MAX_NUMBER_OF_PREDICTIONS ""
ENV MIN_DISTANCE ""
ENV MIN_STATIC_DISTANCE ""
ENV MIN_DETECTIONS ""
ENV IOU ""
ENV FRAMES_SKIP_AFTER_DETECT ""
ENV MIN_DETECTIONS ""

# Run the application
ENTRYPOINT ["python" , "queue_harvesting.py"]