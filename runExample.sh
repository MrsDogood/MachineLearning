#!/usr/bin/env bash

./build.sh || exit 1
mvn exec:java \
    -Dexec.mainClass="com.github.mrsdogood.example.SimpleExample" \
    -Dexec.args="$@" || exit 1
