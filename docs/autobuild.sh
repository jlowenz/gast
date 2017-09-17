#!/usr/bin/env bash

pushd /data
latexmk -silent -pvc -view=none -pdf -lualatex -bibtex lit-review.tex
