# Use an Alpine-based Python 3.10 image for a smaller footprint
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /usr/src/app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# been used for alpine musl-dev
# musl-dev libffi-dev python3-dev


# Copy only necessary files into the container from the docker_playground directory
COPY . /usr/src/app/

# Install Python packages from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Make the start script executable
RUN chmod +x /usr/src/app/start.sh

# Expose port for TensorBoard
EXPOSE 6006

# Set the default command to execute the start script which in turn runs the Python script
CMD ["/usr/src/app/start.sh"]
