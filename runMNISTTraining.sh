#!/usr/bin/env bash

./build.sh || exit 1
args="${@}"
if [ $# -eq 0 ]; then
    mkdir -p dat/
    args="dat/mnist_0.dat 0"
fi
echo "Running with arguments: \"${args}\""
mvn exec:java \
    -Dexec.mainClass="com.github.mrsdogood.example.MNISTTrainer" \
    -Dexec.args="${args}" || exit 1 
