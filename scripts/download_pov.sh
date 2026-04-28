#!/bin/bash
# v2: --no-perms (mfs doesn't allow chown), no set -e (don't bail on errors), retry-safe
SRC=claude_access@74.50.81.94:/root/vla_work/vla_datasets/pov_surgery_data/POV_Surgery_data
DST=/workspace/datasets/POV_Surgery_data
PWD_VAL='claude@1337'
RSYNC_FLAGS="-rlt --no-perms --no-owner --no-group --partial --inplace"

mkdir -p $DST

# Each parallel rsync, log to separate files
for sub in POV_Surgery_info.csv annotation handoccnet_train tool_mesh mask mask_blender color; do
  sshpass -p "$PWD_VAL" rsync $RSYNC_FLAGS -e 'ssh -o StrictHostKeyChecking=no' "$SRC/$sub" "$DST/" \
    > /workspace/logs/pov_${sub//\//_}.log 2>&1 &
done
wait
echo "POV_V2_DONE"
du -sh $DST/*
