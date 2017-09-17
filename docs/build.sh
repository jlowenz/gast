#!/usr/bin/env bash

TARGET=$1
pushd /data
makeglossaries $TARGET
latexmk -pdf -f -xelatex -bibtex $TARGET
