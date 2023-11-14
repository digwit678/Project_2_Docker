#!/bin/bash

# Clone the GitHub repository
echo "Cloning the repository..."
git clone https://github.com/digwit678/Project_2_Docker.git

# Navigate to the docker_playground directory inside the cloned repository
echo "Navigating to the docker_playground directory..."
cd Project_2_Docker/docker_playground

# Build the Docker image
echo "Building the Docker image..."
docker build -t mlops_project_2:playground .

# Run the Docker image
echo "Running the Docker image..."
docker run -p 6006:6006 mlops_project_2:playground

# Indicate completion
echo "Docker image running. Access TensorBoard at port 6006:6006"
