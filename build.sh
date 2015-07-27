#!/usr/bin/env bash

(cd extern/ && make) || exit 1
mvn compile || exit 1
