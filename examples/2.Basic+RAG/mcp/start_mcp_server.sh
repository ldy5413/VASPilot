#!/bin/bash

export PMG_VASP_PSP_DIR=your-path-to-potcars/POTCARS/
nohup vaspilot_mcp --config your-path-to-this-example/configs/mcp_config.yaml --port 8933 > mcp.log 2>&1 &

# get PID for server process
SERVER_PID=$!

# save PID to pid.txt
echo $SERVER_PID > pid.txt

echo "vaspilot_mcp started, PID: $SERVER_PID"
echo "PID saved to pid.txt"
echo "log output redirected to mcp.log"
