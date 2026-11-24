#!/bin/sh
# tools/fork/inkling_release.sh - the fork's release build of strata-inkling for Linux x86_64, CUDA 13 (docs/FORK.md,
# "Releasing from the fork"). Run it inside nvidia/cuda:13.0.2-devel-ubuntu24.04 with the repository at /w/src, for
# example on a Linux box with podman (no toolchain on the host):
#
#   git archive --format=tar HEAD engine tools/packaging terminal/strata_control tools/fork | ssh box 'mkdir -p ~/inkling-release/src && tar -x -C ~/inkling-release/src'
#   podman run --rm -e COMMIT=<short sha> -v ~/inkling-release:/w:Z docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 sh /w/src/tools/fork/inkling_release.sh
#
# It mirrors upstream's package: archs 75;80;86;89;120 (+ compute_120), min driver 580.65, the CUDA runtime, cuBLAS,
# cuBLASLt and libgomp bundled. The unsigned package lands in /w/out/<version>-linux-x86_64-cuda-cuda13; sign and pack it
# with tools/packaging/feed_tool.py sign-package / pack-engine on the machine that holds the key.
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends cmake ninja-build python3 libgomp1 >/dev/null
cd /w/src
archs="75;80;86;89;120"
cmake -S engine --preset cuda-compile -B /w/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES="$archs" -DSTRATA_INK_CUDA_ARCHS="$archs"
cmake --build /w/build --target strata-inkling -j "$(nproc)"
exe=/w/build/src/inkling2/strata-inkling
ver=$(sed -n 's/^#define INKLING_VERSION "\([0-9.]*\)".*/\1/p' engine/src/inkling2/ink_main.cpp)
got=$("$exe" --version | awk '{print $2}')
[ "$got" = "$ver" ] || { echo "the built engine says $got, ink_main.cpp says $ver"; exit 1; }
stage=/w/stage; rm -rf "$stage"; mkdir -p "$stage"
cp "$exe" "$stage/"
cuda=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
echo "{\"version\": \"$ver\", \"shape\": \"inkling2\", \"source\": \"kingdonb/Strata tools/fork/inkling_release.sh\", \"commit\": \"${COMMIT:-unknown}\", \"archs\": \"$archs\", \"cuda\": \"$cuda\", \"portable\": true}" > "$stage/BUILD.json"
python3 tools/packaging/collect_libs.py --binary "$exe" --search /usr/local/cuda/lib64 --search /usr/lib/x86_64-linux-gnu --out "$stage" --also "libcublasLt.so*" --also "libgomp.so*"
rm -rf /w/out; mkdir -p /w/out
python3 tools/packaging/package_engine.py --src "$stage" --backend cuda --version "$ver" --out /w/out \
  --arch sm_75 --arch sm_80 --arch sm_86 --arch sm_89 --arch sm_120 --arch compute_120 \
  --min-driver nvidia=580.65 --variant cuda13 --platform linux-x86_64 --build-id "${COMMIT:-unknown}+cuda$cuda" \
  --include SYSTEM-LIBS.txt --shape inkling2
for d in /w/out/*/; do python3 tools/packaging/package_engine.py --verify "$d"; python3 tools/packaging/binary_deps.py "$d"; done
ls -la /w/out/*/
