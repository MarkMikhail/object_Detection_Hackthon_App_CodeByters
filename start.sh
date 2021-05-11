#!/bin/bash
sudo docker-compose -f edge-service.yml  up -d
docker logs --follow obj_detection