#!/usr/bin/env bash
# Run from root of project

mkdir -p build
cd build || exit

wget https://www.csie.ntu.edu.tw/~cjlin/liblinear/oldfiles/liblinear-1.96.zip
unzip liblinear-1.96.zip
rm liblinear-1.96.zip
cd liblinear-1.96 || exit
make

echo "Successfully installed liblinear-1.96"
echo "Have a good day"