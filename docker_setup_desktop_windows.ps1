# Clone the GitHub repository
Write-Host "Cloning the repository..."
git clone https://github.com/digwit678/Project_2_Docker.git

# Navigate to the project directory
Write-Host "Navigating to the project directory..."
cd Project_2_Docker

# Build the Docker image
Write-Host "Building the Docker image..."
docker build -t project2_docker .

# Run the Docker container
Write-Host "Running the Docker container..."
docker run -p 6006:6006 project2_docker

# Indicate completion
Write-Host "Docker container is running. Access TensorBoard at http://localhost:6006"
