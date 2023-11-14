# Project 2: Docker - Machine Learning Operations

## Overview
This repository contains the Dockerized ML project for the MRPC task using the DistilBERT model. Here we provide detailed instructions for setting up and running the project both locally using Docker Desktop and on Docker Playground.


## Local Setup Using Docker Desktop


## Prerequisites
Before beginning, ensure you have the following installed:
- Docker  (<a href="https://docs.docker.com/get-docker/" target="_blank">Installation Guide</a>)
- Git (<a href="https://git-scm.com/book/en/v2/Getting-Started-Installing-Git" target="_blank">Installation Guide</a>)
  
### Automated Setup (For Both Windows and Linux)
1. Open your terminal (Command Prompt or PowerShell for Windows, Terminal for Linux).
2. Run the setup script for local execution:
```bash
   bash setup_docker_local.sh
```` 
This will run the following bash commands: 

#### a) Clone the Repository
```bash
git clone https://github.com/digwit678/Project_2_Docker.git
````
#### b) Change to Base Folder
```bash
cd Project_2_Docker
````
#### c) Build the Docker Image
```bash
docker build -t project2_docker .
```
#### d) Run the Docker Container
```bash
docker run -p 6006:6006 project2_docker
```
3. Access TensorBoard  
<a href="http://localhost:6006" target="_blank">Open http://localhost:6006</a> to monitor the training progress on TensorBoard. Ensure the port is not used by another service and is not blocked by your firewall. 

## Docker Playground Setup
Execution on Docker Playground is experimental and may face resource constraints.

### Access Docker Playground
<a href="https://labs.play-with-docker.com/" target="_blank">Visit Docker Playground</a>
and start a session.
