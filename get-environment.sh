#!/bin/bash
# Captures host and container environment for troubleshooting.
# Run from the directory where you cloned atlas-voice-training into (e.g. /tmp/atlas-voice-training).

echo "=== Host ==="
uname -srm
grep PRETTY_NAME /etc/os-release
free -h | head -2
lscpu | grep "Model name"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi: not found"
docker --version

echo ""
echo "=== Docker Image ==="
docker image inspect atlas-voice-training --format \
  'Created: {{.Created}}
Size: {{.Size}}
Base layers: {{index .RootFS.Layers 0}}' 2>/dev/null || echo "Image not found"

echo ""
echo "=== Container ==="
docker run --rm --entrypoint bash atlas-voice-training -c '
  ldd --version 2>&1 | head -1
  dpkg -l cuda-cudart* 2>/dev/null | grep ^ii | \
    awk "{print \"cuda-runtime: \" \$2 \" \" \$3}" || \
    echo "CUDA version: unknown"
  python3.10 -c "
import onnxruntime, torch, numpy, scipy
import audiomentations, torch_audiomentations
import flatbuffers, google.protobuf
import os
print(f\"onnxruntime=={onnxruntime.__version__}\")
print(f\"torch=={torch.__version__}\")
print(f\"numpy=={numpy.__version__}\")
print(f\"scipy=={scipy.__version__}\")
print(f\"audiomentations=={audiomentations.__version__}\")
print(f\"torch-audiomentations=={torch_audiomentations.__version__}\")
print(f\"flatbuffers=={flatbuffers.__version__}\")
print(f\"protobuf=={google.protobuf.__version__}\")
print(f\"cpu_count={os.cpu_count()}\")
"
'
