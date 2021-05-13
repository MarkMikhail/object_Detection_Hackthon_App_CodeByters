#!/bin/bash
source /opt/intel/openvino/bin/setupvars.sh
cd main_app/
while true
do
     
     echo "build_ver:1"
     echo "*********************running object detection  app********************************************"
     python3 object_detection.py -i input/obj-detection-old.mp4 -o shared/obj-detection-infered.mp4
     echo "detection finished"
     sleep 1
done     
