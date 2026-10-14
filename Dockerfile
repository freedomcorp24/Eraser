# GramsAI magic eraser: RunPod serverless worker (docs/features/DESIGN_CANVAS_SPEC.md 19). LaMa inpainting on a GPU.
# Same base and torch build as the Z-Image worker (github.com/freedomcorp24/ZImage): CUDA 12.4, so enable only Ampere /
# Ada GPUs on the endpoint (PITFALLS #17: Blackwell cards crash cu124 handlers).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir numpy==1.26.4 pillow==11.3.0 runpod==1.12.0
# WEIGHTS BAKED IN (ARCHITECTURE "Self-hosted media": no download at boot, no network volume pinning one datacenter).
# big-lama.pt is LaMa's big model as TorchScript, the file IOPaint uses, 196 MB, md5 from IOPaint 1.6.0 lama.py.
RUN mkdir -p /models && \
    curl -fsSL -o /models/big-lama.pt https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt && \
    echo "e3aa4aaa15225a33ec84f9f4bc47e500  /models/big-lama.pt" | md5sum -c -
COPY lama.py handler.py verify.py /app/
WORKDIR /app
# Proved at BUILD time: the model loads on the CPU and erases a test square; a broken image never reaches a user.
RUN python3 /app/verify.py /models/big-lama.pt
CMD ["python3", "-u", "/app/handler.py"]
