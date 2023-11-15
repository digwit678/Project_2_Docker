# Check if Docker is running
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker does not seem to be running, please start Docker Desktop, and retry."
    Read-Host "Press ENTER to exit..."
    exit 1
}

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
