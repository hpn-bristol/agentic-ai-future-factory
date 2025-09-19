#!/bin/bash
gunicorn -b :8989 -t ${TIMEOUT:=300} \
  -k uvicorn.workers.UvicornWorker \
  main:app
