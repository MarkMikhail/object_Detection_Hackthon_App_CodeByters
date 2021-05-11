#!/bin/bash
sudo docker-compose -f edge-service.yml  down
sudo ./build.sh
sudo docker-compose -f edge-service.yml  up -d
docker logs --follow obj_detection