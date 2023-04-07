#!/bin/bash

echo "Start NSFW API"

sudo pip3 install -r requirement.txt
sudo flask run --host=0.0.0.0 --port=5002
