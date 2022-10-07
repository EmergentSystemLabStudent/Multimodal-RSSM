docker run --gpus '"device=0"' -it --shm-size 32G --privileged -p 0888:8888 -p 7900:5900 -p 8006:6006 -p 0265:8265 -p 7000:5000 -v ~/ws/sharespace:/root/sharespace --rm $1 
