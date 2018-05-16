#!/usr/bin/env bash

BASE_DIR=$(pwd)

CMD=${@:-"/bin/bash"}
docker run --rm -it \
       -v $BASE_DIR:/data \
       -v $BASE_DIR/../src:/src \
       -e LOCAL_USER_ID=$(id -u) \
       -e LOCAL_GROUP_ID=$(id -g) \
       docker-ict.di2e.net/arl-latex:1.3 $CMD
