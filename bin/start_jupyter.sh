#!/bin/bash

FILE=~/.bashrc
if [ -f "$FILE" ]; then
  source ~/.bashrc
fi
FILE=~/.profile
if [ -f "$FILE" ]; then
  source ~/.profile
fi
FILE=~/.bash_profile
if [ -f "$FILE" ]; then
  source ~/.bash_profile
fi
cd .. && source ~/.miniconda3/bin/activate mri_project && jupyter notebook