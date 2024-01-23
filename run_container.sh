#!/bin/bash

sudo docker build --no-cache -t lgn6d .
thispid=$(sudo docker run --gpus 0 --network=host --name=lgn6d_gpu -t -d -v /home/stefan:/stefan lgn6d)

#sudo nvidia-docker exec -it $thispid bash

#sudo nvidia-docker container kill $thispid
#sudo nvidia-docker container rm $thispid
