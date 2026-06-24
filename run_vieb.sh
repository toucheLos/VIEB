#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python user_interface.py "$@"
