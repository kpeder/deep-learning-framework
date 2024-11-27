#!/bin/bash

echo $(pwd)

if [ -z "${PYTHONPATH}" ]; then
    echo "PYTHONPATH is unset or set to the empty string"
    export PYTHONPATH=$(pwd)/src
fi

echo PYTHONPATH=$PYTHONPATH

if [ -z "${MODELPY}" ]; then
    echo "MODELPY is unset or set to the empty string"
    exit 1
fi

python3 $MODELPY



