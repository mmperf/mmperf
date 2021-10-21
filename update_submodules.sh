#!/bin/bash -x

COMMIT_PUSH=false

while getopts “p” OPTION
do
     case $OPTION in
         p)
             echo "Pushing changes up.."
             COMMIT_PUSH=true
             ;;
         ?)
             echo "Unsupported option.. -p for pushing changes up after update"
             exit
             ;;
     esac
done

echo "Updating repos.."
#repos with master branches
for master_branch in flatbuffers cpuinfo
do
    echo Update master branch for ... $master_branch
    cd external/$master_branch && git fetch --all && git checkout origin/master && cd -
done

#repos with master branches
for main_branch in benchmark iree  iree-llvm-sandbox  llvm-project tvm Halide
do
    echo Update master branch for ... $main_branch
    cd external/$main_branch && git fetch --all && git checkout origin/main && cd -
done

if [ "$COMMIT_PUSH" = true ]; then
  echo "Checking out transformer-benchmarks..."
  git add .
  git commit -m "Roll external deps"
  echo git push https://github.com/mmperf/mmperf
fi
