#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [ -f "/root/netem_ws/install/setup.bash" ]; then
  source /root/netem_ws/install/setup.bash
fi

exec "$@"
