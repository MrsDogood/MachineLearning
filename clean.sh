#!/usr/bin/env bash

if [[ $1 == "--all" ]]; then
    echo -n "Are you sure you want to clean ALL data, "
    echo -n "including training data and saved neural nets? [N/y] "
    read confirm
    if [[ $confirm == "y" ]]; then
        echo "Cleaning external dependencies..."
        (cd extern/ && make clean) || exit 1
        echo "Cleaning training and neural net data..."
        rm -rf dat/
        echo "Cleaning maven build..."
        rm -rf target/
        exit 0
    else
        echo "Cleaning Canceled"
    fi
else
    echo "Cleaning maven build..."
    rm -rf target/
fi
