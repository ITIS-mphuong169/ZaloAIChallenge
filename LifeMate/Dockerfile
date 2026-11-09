# LifeMate - speech model + language model + agent. Runs offline on one GPU.
#
# STATUS: not built yet. The image needs two model folders that only exist after the steps of ../README.md section 5:
#   models/asr-final   the fine-tuned speech model, Hugging Face format (section 5.7)
#   models/llm         the language model weights, Hugging Face format (section 6.3 tells how to download them)
# Build from the LifeMate folder:   docker build -t lifemate:v1 .
#
# The base image already holds CUDA, torch, transformers and the vLLM server, so nothing is installed from the network.
# Check the organisers' Docker guideline (base image, GPU, size limit) before the final submission.
FROM vllm/vllm-openai:v0.33.0
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /app
COPY lifemate /app/lifemate
# solved train samples: shown to the language model as examples
COPY ["Dataset/extracted/train+public_test/train/train.jsonl", "/app/train/train.jsonl"]
COPY models/asr-final /models/asr
COPY models/llm /models/llm
COPY predict.sh /app/predict.sh
RUN chmod +x /app/predict.sh && mkdir -p /data /result
ENTRYPOINT ["/app/predict.sh"]
