# Michael-J.-Mauboussin-Twin

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest --model Qwen/Qwen2-VL-2B-Instruct --task generate \
  --trust-remote-code --limit-mm-per-prompt image
```