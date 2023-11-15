# Project 2: Docker - Machine Learning Operations

## Overview
This repository contains the Dockerized ML project for the MRPC task using the DistilBERT model. Instructions are provided for setting up and running the project automatically or manually both locally using Docker Desktop and on Docker Playground .

## Prerequisites
- Docker  (<a href="https://docs.docker.com/get-docker/" target="_blank">Installation Guide</a>)
- Git (<a href="https://git-scm.com/book/en/v2/Getting-Started-Installing-Git" target="_blank">Installation Guide</a>)

## Docker Desktop Setup
The repository includes two scripts for automating the setup process:
- `docker_setup_desktop_linux.sh` for Linux users
- `docker_setup_desktop_windows.ps1` for Windows users

These scripts handle cloning the repository, building the Docker image, and running the Docker container.

### Folder Structure
The base folder of the GitHub repository includes:
- `README.md`: Documentation file with setup instructions.
- `requirements.txt`: List of Python packages required for the project.
- `start.sh`: Script to start the training and TensorBoard logging.
- `docker_playground/`: Directory with files necessary for running the project in Docker Playground.
- `lightning_logs/`: Directory where TensorBoard logs will be stored.
- `DistilBERT_MRPC_Script.py`: Python script for training the DistilBERT model.
- `docker_setup_desktop_linux.sh`: Bash script to automate the setup process on Linux.
- `docker_setup_desktop_windows.ps1`: PowerShell script to automate the setup process on Windows.
- `Dockerfile`: Configuration file for creating the Docker image.

### Automated Setup Using Docker Desktop
1. Choose the appropriate `docker_setup_desktop_[linux or windows].sh` script based on your operating system and download it directly from github by: <br></br>
   a) Click the chosen shell/powershell script in repository main folder\
   b) Navigate to the upper right corner below your account symbol + open "More file actions"/"..." menu\
   c) Chose *Download* to only load the selected setup script.
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
     **Right-click** on `docker_setup_desktop_windows.ps1` file and select **Run with PowerShell**.

6. Follow the prompts in the terminal to complete the setup.

7. After running the setup script, the Docker container will start, and you can <a href="http://localhost:6006" target="_blank">access TensorBoard at http://localhost:6006</a> to monitor the training progress. <br></br>  **Ensure the port 6006 is not used by another service and is not blocked by your firewall** (For instructions on how to change the tensoboard logging port pls refer to the ***Troubleshooting*** section below).

### Manual Setup Using Docker Desktop
1. Open a terminal.
2. Execute the following commands:<br></br>
   a) Clone the repository
     ```bash
     git clone https://github.com/digwit678/Project_2_Docker.git
     ````
   b) Navigate to the repository directory
      ```bash
      cd Project_2_Docker
      ````
   c) Build the Docker image
      ```bash
      docker build -t project2_docker .
      ```
   d) Run the Docker container
      ```bash
      docker run -p 6006:6006 project2_docker
      ```  
   e) <a href="http://localhost:6006" target="_blank">Access TensorBoard at http://localhost:6006</a>

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
For Docker Playground, I have optimized the setup to work within the resource constraints typically found in such environments. The `docker_playground` directory contains only the essential files needed to run the training script on the MRPC task using the DistilBERT model.  

### `docker_playground` Directory Contents
- `msr_paraphrase_test.txt` and `msr_paraphrase_train.txt`: These text files contain the MRPC dataset used for training and evaluating the model.
- `requirements.txt`: A list of Python packages required for running the project. This file has been optimized to exclude unnecessary packages to save memory.
- `start.sh`: A shell script that is used to start TensorBoard and execute the Python training script.
- `Task_3_DistilBERT_MRPC_Script.py`: The Python script that conducts the training of the DistilBERT model. It has been adapted to work with local text files instead of using the datasets library.
- `lightning_logs`: A directory that TensorBoard uses to log training progress.
- `docker_setup_playground.sh`: A shell script that automates the process of setting up and running the Docker container in Docker Playground.
- `Dockerfile`: A Dockerfile configured specifically for Docker Playground, with adjustments made to work within the platform's resource limitations.

### Running the Project in Docker Playground

### Access Docker Playground
<a href="https://labs.play-with-docker.com/" target="_blank">Visit Docker Playground</a>
and start a session.

### Steps for Docker Playground
Chose either to set it up automatically by using the setup script or follow the steps in the manual setup process. 
#### Automated Setup Using Docker Playground 
1. Drag and drop `docker_setup_playground.sh` found in `Project_2_Docker/docker_playground/` directory into the docker playground shell to upload it
2. Run the automated setup script in the current sessions shell:  
    ```bash
      sh docker_setup_playground.sh
    ```
#### Manual Setup Using Docker Desktop  
1. Clone the repository and navigate to the `docker_playground` directory:
   ```bash
   git clone https://github.com/digwit678/Project_2_Docker.git
   cd Project_2_Docker/docker_playground
    ```
2. Build the Docker image
      ```bash
      docker build -t project2_docker .
      ```
3. Run the Docker container
      ```bash
      docker run -p 6006:6006 project2_docker
      ```  
