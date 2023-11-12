# Use an official Python 3.10 runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for TensorBoard
EXPOSE 6006

# Copy the start script into the container and make it executable
COPY start.sh /usr/src/app/start.sh
RUN chmod +x /usr/src/app/start.sh

# Set the default command to execute the Python script
CMD ["/usr/src/app/start.sh"]

