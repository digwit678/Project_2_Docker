# Project 2: Docker - Machine Learning Operations

## Overview
This repository contains the Dockerized ML project for the MRPC task using the DistilBERT model. Instructions are provided for setting up and running the project both locally using Docker Desktop and on Docker Playground.

## Prerequisites
- Docker  (<a href="https://docs.docker.com/get-docker/" target="_blank">Installation Guide</a>)
- Git (<a href="https://git-scm.com/book/en/v2/Getting-Started-Installing-Git" target="_blank">Installation Guide</a>)

## Automated Setup Using Docker Desktop
The repository includes two scripts for automating the setup process:
- `docker_setup_desktop_linux.sh` for Linux users
- `docker_setup_desktop_windows.ps1` for Windows users

These scripts handle cloning the repository, building the Docker image, and running the Docker container.

### Folder Structure
The base folder of the GitHub repository includes:
- `README.md`: Documentation file with setup instructions.
- `requirements.txt`: List of Python packages required for the project.
- `start.sh`: Script to start the training and TensorBoard logging.
- `.git`: Hidden folder containing Git version control history.
- `docker_playground/`: Directory with files necessary for running the project in Docker Playground.
- `lightning_logs/`: Directory where TensorBoard logs will be stored.
- `DistilBERT_MRPC_Script.py`: Python script for training the DistilBERT model.
- `docker_setup_desktop_linux.sh`: Bash script to automate the setup process on Linux.
- `docker_setup_desktop_windows.ps1`: PowerShell script to automate the setup process on Windows.
- `Dockerfile`: Configuration file for creating the Docker image.

### Executing the Setup Script
1. Choose the appropriate docker_setup_desktop_[linux or windows] script based on your operating system.
2. Open a terminal (PowerShell for Windows, Terminal for Linux).
3. Make sure Docker is running
4. Navigate to the directory where you want to clone the repository.
5. Execute the setup script:
   - For Linux:
     ```
     bash docker_setup_desktop_linux.sh
     ```
   - For Windows:
     Right-click on `docker_setup_desktop_windows.ps1` and select "Run with PowerShell".

6. Follow the prompts in the terminal to complete the setup.

7. After running the setup script, the Docker container will start, and you can <a href="http://localhost:6006" target="_blank">access TensorBoard at http://localhost:6006</a> to monitor the training progress. **Ensure the port 6006 is not used by another service and is not blocked by your firewall**.  
For instructions on how to change the tensoboard logging port pls refer to the ***Troubleshooting*** section below. 

### Manual Setup Steps
1. Open a terminal.
2. Execute the following commands in order:
   - Clone the repository
   - Navigate to the repository directory
   - Build the Docker image
   - Run the Docker container
   - Access TensorBoard

### Troubleshooting
- Docker Build Fails: Ensure Docker is running and you have internet connectivity.
- TensorBoard Not Accessible: Check Docker container status and port mapping.
- Change TensorBoard Port: If port 6006 cannot be used, update the Dockerfile and start.sh to use a different port.

## Docker Playground Setup
Note: Running on Docker Playground is experimental and may face resource constraints.

### Access Docker Playground
Visit Docker Playground and start a session.

### Steps for Docker Playground
1. Clone the repository in the Docker Playground terminal.
2. Build the Docker image.
3. Run the Docker container.
4. Click the "6006" link in the Playground UI to view TensorBoard.
5. The script execution in Playground may face interruptions due to resource limits.
