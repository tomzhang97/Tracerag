# Local LLM Configuration Guide

This guide explains how to use TraceRAG with local LLMs (vLLM or Ollama) instead of hosted APIs.

## Why Local LLMs?

- **Cost**: No API fees
- **Privacy**: Data stays on your infrastructure
- **Customization**: Fine-tune models for engineering domains
- **Latency**: Potentially faster for local deployments

## Option 1: vLLM

### Installation

```bash
# Install vLLM
pip install vllm

# Download a model (e.g., Llama-3-8B)
# Models are automatically downloaded from HuggingFace
```

### Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

### Configure TraceRAG

Edit `tracerag/config/defaults.yaml`:

```yaml
llm:
  model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"  # vLLM doesn't require API key
  temperature: 0.1
  max_tokens: 2000
```

### Test

```bash
tracerag-search \
  --index_root /data/tracerag/index \
  --query "What is valve V-101?"
```

---

## Option 2: Ollama

### Installation

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3
```

### Start Ollama Server

Ollama runs as a service automatically. Default endpoint: `http://localhost:11434`

### Configure TraceRAG

Edit `tracerag/config/defaults.yaml`:

```yaml
llm:
  model_name: "llama3"  # or "llama3:70b" for larger model
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"  # Ollama expects any non-empty key
  temperature: 0.1
  max_tokens: 2000
```

### Test

```bash
tracerag-search \
  --index_root /data/tracerag/index \
  --query "What is the torque for bolt CFG-AX14-K2?"
```

---

## Option 3: Remote vLLM (Multi-GPU Server)

For production deployments with high throughput:

### Server Setup

```bash
# On GPU server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \  # 4 GPUs
  --host 0.0.0.0 \
  --port 8000
```

### Client Configuration

```yaml
llm:
  model_name: "meta-llama/Meta-Llama-3-70B-Instruct"
  base_url: "http://gpu-server.company.com:8000/v1"
  api_key: "EMPTY"
  temperature: 0.1
  max_tokens: 2000
```

---

## Recommended Models for Engineering QA

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **Llama-3-8B** | 8B | Fast | Good | Development, testing |
| **Llama-3-70B** | 70B | Medium | Excellent | Production |
| **Mistral-7B** | 7B | Fast | Good | Quick queries |
| **Qwen2-7B** | 7B | Fast | Very Good | Technical documents |
| **CodeLlama-34B** | 34B | Medium | Excellent | Code/diagram heavy |

---

## Performance Tuning

### vLLM Options

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-size 2 \      # Use 2 GPUs
  --max-model-len 4096 \           # Context length
  --gpu-memory-utilization 0.9 \   # Use 90% of GPU memory
  --dtype half                      # FP16 for speed
```

### Ollama Options

```bash
# Set context length
ollama run llama3 --ctx-length 4096

# Set number of threads
ollama run llama3 --threads 8
```

---

## Benchmarking

### Test Latency

```bash
time python -m tracerag.cli.demo_search \
  --index_root /data/tracerag/index \
  --query "What is valve V-101?"
```

### Run Full Evaluation

```bash
tracerag-eval \
  --benchmark eng_bench \
  --index_root /data/tracerag/index \
  --data_path examples/example_eng_bench.json \
  --output_file results.json
```

---

## Troubleshooting

### Issue: vLLM server not responding

**Solution**: Check if server is running and accessible:
```bash
curl http://localhost:8000/v1/models
```

### Issue: Out of memory errors

**Solution**: Reduce model size or increase GPU memory:
```bash
--max-model-len 2048  # Reduce context length
--gpu-memory-utilization 0.8  # Reduce memory usage
```

### Issue: Slow responses

**Solution**: Use tensor parallelism or smaller model:
```bash
--tensor-parallel-size 2  # Split across 2 GPUs
```

### Issue: Ollama connection refused

**Solution**: Restart Ollama service:
```bash
ollama serve
```

---

## Cost Comparison

| Setup | Cost per 1M tokens | Latency (avg) | Hardware Required |
|-------|-------------------|---------------|-------------------|
| **OpenAI GPT-4** | $30 | 2-5s | None |
| **Anthropic Claude** | $15 | 1-3s | None |
| **vLLM (8B, local)** | $0 | 0.5-1s | 1x RTX 3090 |
| **vLLM (70B, server)** | $0* | 1-2s | 4x A100 |
| **Ollama (8B)** | $0 | 1-2s | 1x RTX 3090 |

*Electricity costs apply

---

## Production Checklist

- [ ] vLLM/Ollama server running and accessible
- [ ] Model downloaded and tested
- [ ] `base_url` configured in `defaults.yaml`
- [ ] Firewall allows access to LLM server
- [ ] Benchmark run confirms acceptable accuracy
- [ ] Load testing completed for expected QPS
- [ ] Monitoring/logging configured
- [ ] Backup OpenAI/Anthropic API key for fallback

---

## Next Steps

- **Fine-tuning**: Fine-tune Llama-3 on your engineering corpus for better accuracy
- **Caching**: Implement response caching for common queries
- **Load Balancing**: Use multiple vLLM instances behind a load balancer
- **Quantization**: Use 4-bit quantization (GPTQ/AWQ) for faster inference

---

For more help, see:
- vLLM docs: https://docs.vllm.ai/
- Ollama docs: https://ollama.com/docs
- TraceRAG issues: https://github.com/your-org/tracerag/issues
