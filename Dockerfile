# Use an official Python runtime based on Debian Bookworm (stable)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV and other vision libraries
# This is crucial for video/camera access
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy the Python script and virtual environment's requirements
COPY main.py ./
COPY .venv/ .venv/

# Install the dependencies from the virtual environment
RUN /app/.venv/bin/pip install --no-cache-dir -r ./.venv/requirements.txt opencv-python

# Make sure the user inside the container has access to devices
# This is a security measure and is key to camera access
# We create a new user to avoid running as root
RUN groupadd -g 1000 appgroup && useradd -u 1000 -g 1000 appuser
USER appuser

# This command will be executed when the container starts
CMD ["/app/.venv/bin/python", "main.py"]