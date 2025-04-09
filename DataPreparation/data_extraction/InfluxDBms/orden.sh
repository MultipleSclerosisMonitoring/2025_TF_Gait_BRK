#!/bin/bash
pip3 install -r requirements.txt --break-system-packages
pip wheel . --no-cache-dir --disable-pip-version-check --no-deps --wheel-dir dist --no-build-isolation
sudo pip3 install dist/InfluxDBms-0.1.1-py3-none-any.whl --break-system-packages
