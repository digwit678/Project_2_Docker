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
1. Choose the appropriate `docker_setup_desktop_[linux or windows].sh` script based on your operating system and download it directly from github by:
   a.) click on the chosen shell/powershell script in repository main folder
   b.) navigate to the upper right corner below your account symbol + open "More file actions" menu "..."
   c.) Chose *Download* to only load the setup script. 
2. Open a terminal (PowerShell for Windows, Terminal for Linux).
3. Make sure Docker is running
4. Navigate to the directory where you want to clone the repository.
5. Execute the setup script:
   - For Linux:
<br></br>
     ```
     bash docker_setup_desktop_linux.sh
     ```
   - For Windows: <br></br>
     <br></br> 
     Right-click on `docker_setup_desktop_windows.ps1` and select "Run with PowerShell".

6. Follow the prompts in the terminal to complete the setup.

7. After running the setup script, the Docker container will start, and you can <a href="http://localhost:6006" target="_blank">access TensorBoard at http://localhost:6006</a> to monitor the training progress. **Ensure the port 6006 is not used by another service and is not blocked by your firewall** (For instructions on how to change the tensoboard logging port pls refer to the ***Troubleshooting*** section below).

### Manual Setup Steps
1. Open a terminal.
2. Execute the following commands in order:
   - Clone the repository
     ```bash
     git clone https://github.com/digwit678/Project_2_Docker.git
     ````
   - Navigate to the repository directory
      ```bash
      cd Project_2_Docker
      ````
   - Build the Docker image
      ```bash
      docker build -t project2_docker .
      ```
   - Run the Docker container
      ```bash
      docker run -p 6006:6006 project2_docker
      ```  
   - <a href="http://localhost:6006" target="_blank">Access TensorBoard at http://localhost:6006</a>

### Troubleshooting
- Docker Build Fails: Ensure Docker is running and you have internet connectivity.
- TensorBoard Not Accessible: Check Docker container status and port mapping.
- Change TensorBoard Port: If port 6006 cannot be used, ***update the port mapping*** in the Dockerfile (docker run -p ***6006:6006*** project2_docker) and start.sh (tensorboard --logdir=/usr/src/app/lightning_logs --port=***6006*** --bind_all &) to use a different port.
- Error when running automated (Windows) powershell setup script: '/docker_setup_desktop_windows.ps1' is not recognized as an internal or external command
  
a) Check Execution Policy 
```bash
Get-ExecutionPolicy
```  
If the policy is set to *Restricted*, you will need to change it to allow script execution. 

b) Change Execution Policy
```bash
Set-ExecutionPolicy RemoteSigned
```  
## Docker Playground Setup
Note: Running on Docker Playground is experimental and may face resource constraints.

### Access Docker Playground
<a href="https://labs.play-with-docker.com/" target="_blank">Visit Docker Playground</a>
and start a session.

### Steps for Docker Playground
#### Automated Setup Using Docker Desktop  
1. Drag and drop `docker_setup_playground.sh` found in `Project_2_Docker/docker_playground/` directory into the docker playground shell to upload it
2. Run the automated setup script in the current sessions shell:  
    ```bash
      sh docker_setup_playground.sh
    ```
