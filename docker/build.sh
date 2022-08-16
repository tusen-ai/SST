#!/bin/sh

# Build docker image
docker build -f Dockerfile -t mmdetection3d ..

# Convert to singularity
singularity build mmdetection3d.sif docker-daemon://mmdetection3d:latest
