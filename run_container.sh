#!/bin/bash

# Run the container with custom flags:

# --name: name of the container
# --cpus: number of cpus to use
# --gpus: gpu devices to use
# --ipc=host: use the host's ipc namespace
# --network=host: use the host's network namespace
# -v /etc/passwd:/etc/passwd:ro: mount the passwd file to the container
# -v /etc/group:/etc/group:ro: mount the group file to the container
# -v /home/user/path/to/repo/MuTILs_Panoptic:/home/mtageld/Desktop/MuTILs_Panoptic:
#          mount the MuTILs_Panoptic repo to the container (Change this to your repo path!)
# -v /data:/data:
#          mount the data directory to the container (Change this to your data path!)
# -v /mnt:/mnt:
#          mount the mnt directory to the container (Change this to your mnt path!)
# --ulimit core=0: disable core dumps
# --rm: remove the container after it exits
# -it: run the container in interactive mode
# --user root: run the container as root
# kheffah/ctme:latest: the name of the image to run
# bash -c "source /home/mtageld/Desktop/MuTILs_Panoptic/.bashrc && bash": the command to run in the container
#          (source the .bashrc file with proper Python paths and start a bash shell)

docker run --name MutilsDev1 \
  --cpus 8 \
  --gpus '"device=1"' \
  --ipc=host \
  --network=host \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /home/lit7907/repos/MuTILs_Panoptic:/home/mtageld/Desktop/MuTILs_Panoptic \
  -v /data:/data \
  -v /mnt:/mnt \
  --ulimit core=0 \
  --rm \
  -it \
  --user root \
  kheffah/ctme:latest \
  bash -c "source /home/mtageld/Desktop/MuTILs_Panoptic/.bashrc && bash"