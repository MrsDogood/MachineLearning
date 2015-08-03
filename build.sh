#!/usr/bin/env bash

echo "Building external non-maven dependencies..."
(cd extern/ && make) || exit 1

echo "Building maven pom..."
mvn compile || exit 1
