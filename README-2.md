NOTES
-----

```bash
echo "deb http://apt.llvm.org/DISTRO_NAME/ llvm-toolchain-DISTRO_NAME main" >> /etc/apt/sources.list
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - 
apt-get update && apt-get upgrade -y
apt-get install -y clang-11 clang-tools-11 libc++1-11 libc++-11-dev \
    libc++abi1-11 libc++abi-11-dev libclang1-11 libclang-11-dev \
    libclang-common-11-dev libclang-cpp11 libclang-cpp11-dev liblld-11 \
    liblld-11-dev liblldb-11 liblldb-11-dev libllvm11 libomp-11-dev \
    libomp5-11 lld-11 lldb-11 llvm-11 llvm-11-dev llvm-11-runtime llvm-11-tools libfuzzer-11-dev
```
