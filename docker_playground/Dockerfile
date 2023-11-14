# Use an Alpine-based Python 3.10 image for a smaller footprint
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Python packages
# Copy the requirements file into the container
COPY requirements.txt .
# Install packages from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files into the container from the current directory
COPY . /usr/src/app/

# Make the start script executable
RUN chmod +x /usr/src/app/start.sh

# Expose port for TensorBoard
EXPOSE 6006

# Set the default command to execute the start script which in turn runs the Python script
CMD ["/usr/src/app/start.sh"]
