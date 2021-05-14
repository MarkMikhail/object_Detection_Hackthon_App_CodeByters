#!/bin/bash
docker container commit object_detection object_detection:latest
docker image tag object_detection:latest localhost:5000/object_detection:latest
docker image push localhost:5000/object_detection:latest