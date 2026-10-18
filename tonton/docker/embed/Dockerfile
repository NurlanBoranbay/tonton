# EmbeddingGemma 2 runtime: torch+CUDA from the local vLLM image, newer HF libs on top.
FROM vllm/vllm-openai:latest
RUN pip install --no-cache-dir -U "transformers>=5.18" "sentence-transformers>=6.1.0" pillow
ENTRYPOINT ["python3"]
