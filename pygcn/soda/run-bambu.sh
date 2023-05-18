#!/bin/bash

set -e
set -o pipefail

if [ -d output/$1 ] 
then
    rm -rf output/$1
    mkdir -p output/$1
else
    mkdir -p output/$1
fi

# PLATFORM=asap7
# DEVICE=asap7-BC
# DEVICE=asap7-TC
# DEVICE=asap7-WC

PLATFORM=nangate45
DEVICE=nangate45

cp output/05$1.ll output/$1/input.ll
cp forward_kernel_test.xml output/$1/forward_kernel_test.xml

pushd output/$1;

docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
opt -O2 -strip-debug input.ll \
  -S -o visualize.ll

docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
bambu -v3 --print-dot \
  -lm --soft-float \
--compiler=I386_CLANG12  \
--device=${DEVICE} \
--clock-period=5 --no-iob \
--experimental-setup=BAMBU-BALANCED-MP \
--channels-number=2 \
--memory-allocation-policy=ALL_BRAM \
--disable-function-proxy \
--generate-tb=forward_kernel_test.xml \
--simulate --simulator=VERILATOR \
--top-fname=forward_kernel \
input.ll 2>&1 | tee bambu-log

popd

# Patch nangate
SCRIPTDIR=scripts
BAMBUDIR=output/$1
KERNELNAME=forward_kernel
source ${SCRIPTDIR}/patch_openroad_synt.sh
