# Test Progress of LogQuantKV on LongBench

## Test round 1

### Test Settings

* Methods: [KiVi, StreamingQuant, LogQuant, PartialStreamingQuant, PartialLogQuant]
* N-bits: [2]
* Full Precision length in [128, 192, 256]

#### Models

##### llama

- [x] meta-llama/Meta-Llama-3-8B-Instruct
- [x] meta-llama/Meta-Llama-3.1-8B-Instruct

#### Qwen

- [ ] Qwen/Qwen1.5-1.8B-Chat
- [ ] Qwen/Qwen1.5-1.8B-Chat-AWQ
- [ ] Qwen/Qwen1.5-4B-Chat
- [ ] Qwen/Qwen1.5-4B-Chat-AWQ
- [x] Qwen/Qwen1.5-7B-Chat
- [x] Qwen/Qwen1.5-7B-Chat-AWQ
- [x] Qwen/Qwen1.5-14B-Chat
- [ ] Qwen/Qwen1.5-14B-Chat-AWQ
- [ ] Qwen/Qwen1.5-32B-Chat
- [ ] Qwen/Qwen1.5-32B-Chat-AWQ
- [ ] Qwen/Qwen2-1.5B-Instruct
- [ ] Qwen/Qwen2-7B-Instruct

##### Phi-3

- [ ] microsoft/Phi-3-mini-128k-instruct
- [ ] microsoft/Phi-3-small-128k-instruct
- [ ] microsoft/Phi-3-medium-128k-instruct