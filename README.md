# Introduction
    This document describes how to build,run,test computer vision edge application on local machine 
# Pre-requisites
1.	docker framework 
 Command to check your docker version
  docker --version 
  Note: Make sure your version number is greater than 19, although it shoudl also work for later version of version 18   

2.	docker compose framework 
docker-compose --version 
 Note: Make sure your version number is greater than 1.26, although it shoudl also work for later version of version 1.28  

3.	Python 3.7 or higher 
python --version 

4.	Zip file that came with the hackathon package named "Object_Detection_Hackthon_App"
5.  VS code (or any other editor) 

# WHat is in the Zip file     
    1. main_app directory : This directoory consists of following subfolders and files
             a.Input directory : This directory consists of input videos to be used for this exercise  
             b.Models directory : This directiry hasd pre-built Computer Vision model, primarily the .bin and .xml file.  
             c.src directory : model libraries /dependencies 
             d.coco_lables.txt : object lables for detection 
             e.object_detection.py : Inference script
    2. Outputs directory : Processed result lands here (You can see resluts of inferencing in)
    3. build.sh : Script for building docker container 
    4. Dockerfile : Dockerfile needed to  build container 
    5. run.sh : Script for initizing the environment and to start inferencing   
    6. edge-service.yml: Docker compose file to start/stop  docker container 

# Now lets follow following steps to build , run ,test computer vision edge app . 

# Step -1 : Unzip file named Object_Detection_Hackthon_App

# Step-2 : Building docker container 

A. Before building docker container you can change image if you wish. If you decided to change name . Open extracted file in VS code editor , Go to build.sh . In  build.sh you will find below line of code 
```

docker build . -t edge/cv/object_detection:hackathon_test

```
You can specify name of your container and tag name here

```
docker build . -t edge/cv/<dontainer_name:tag_name>

```
Save file . Simillary you need to change name in edge-services.yml . Please replace name in below line in edge-services.yml

```
image: edge/cv/<replece container name and tag name here  saperated by  ':'  >

```
Save edge-services.yml file . Now you are ready to build .  Open command line and make sure that you in correct path . Hit below command. 

```
$ cd Object_Detection_Hackthon_App
$ sudo ./build.sh


```
After sucessful build you will see below result 
```
Successfully built 7f92ebbae3f2
Successfully tagged edge/cv/object_detection:hackathon_test
```
After hitting below command you can see newly buit conatiner image 

```
docker images
```

# Step-3 : Running  docker container 
After successful image building and cross checking  we can run container . To run container run below command . 

```
.$ sudo ./docker-compose -f edge-services.yml  up -d 

```

You will see below output on successful start of application 

```
Creating network "object_detection_hackthon_app_default" with the default driver
Creating obj_detection ... done

```
You can varify using below command 

```
$ docker ps
```

# Step-4: Testing output 

You can check procesed result images in /var/camera/images directory . 
**NOTE : AFTER CHECKING IMAGES PLEASE DELETE IMAGES TO AVOID MEMORY SPACE COSUMPTION IF YOU ARE STRUGGING WITH SPACE ISSU**

# Step -5 : Stoppig application 

Using below command you can stop the application 

```
$ sudo ./docker-compose -f edge-services.yml  down

```
Step-6 : Chaning input video (optional)

If you want to use video of your choice for inferencing purpose  , You need to follow below steps . 
A. Place video in input directory 
B.Replace video file name  in run.sh file  . See below example 
```
python3 object_detection.py -i input/<add video file name here>

```
Save file 

**Next follow Step 2, 3,4,5 again**
. 

**NOTE :
1. Sometimes if you are not able to see final output  , Simply stop and start application again 
2. To save output image to your custom path you need to change following in edge-services.yml
```
- "<add your path here >:/main_app/shared/"
```
3. The accuracy may vary due frame quality , camera angle etc factors . 


**Thank You ......!**