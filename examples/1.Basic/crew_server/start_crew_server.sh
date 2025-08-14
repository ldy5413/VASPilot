#!/bin/bash

# start vaspilot_server in background

nohup vaspilot_server --config "your-path-to-this-example/configs/crew_config.yaml" \
    --port 51293 --work-dir "your-path-to-this-example/crew_server/work" --allow-path "your-path-to-this-example/" > server.log 2>&1 &

# get PID for server process
SERVER_PID=$!

# save PID to pid.txt
echo $SERVER_PID > pid.txt

echo "vaspilot_server started, PID: $SERVER_PID"
echo "PID saved to pid.txt"
echo "log output redirected to server.log"
