#!/bin/bash

file="$1"
file+=".tex"

if [ -f $file ]
then
    pdflatex $file

    rm $1.aux
    rm $1.log
else
    echo "File "$file" does not exist"
    exit
fi


