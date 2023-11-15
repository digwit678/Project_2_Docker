#!/bin/bash

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Docker does not seem to be running, please start Docker Desktop, and retry."
    echo "Press enter to exit..."
    read
    exit 1
fi

echo "Cloning the repository..."
git clone https://github.com/digwit678/Project_2_Docker.git

echo "Navigating to the project directory..."
cd Project_2_Docker

echo "Building the Docker image..."
docker build -t project2_docker .

echo "Running the Docker container..."
docker run -p 6006:6006 project2_docker

echo "Docker container is running. Access TensorBoard at http://localhost:6006"
