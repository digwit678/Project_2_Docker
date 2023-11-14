# MLOPS Project 2: Docker

## Overview
This repository hosts the Dockerized ML project focusing on the MRPC task using DistilBERT. Instructions are provided for both local setup and Docker Playground execution.

## Prerequisites
- Docker ([Installation Guide](https://docs.docker.com/get-docker/))
- Git ([Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))

## Local Setup 

### Clone the Repository 

```bash
git clone https://github.com/digwit678/Project_2_Docker.git
````

### Change to Base Folder 
```bash
cd Project_2_Docker
````
## Manual Setup
### Clone the Repository
```bash
git clone https://github.com/digwit678/Project_2_Docker.git
````
### Change to Base Folder
```bash
cd Project_2_Docker
````
### Build the Docker Image
```bash
docker build -t project2_docker .
```
### Run the Docker Container
```bash
docker run -p 6006:6006 project2_docker
```
### Access TensorBoard

Open http://localhost:6006 to monitor the training progress on TensorBoard.

## Docker Playground Setup
Execution on Docker Playground is experimental and may face resource constraints.

### Access Docker Playground
Visit Docker Playground and start a session.
