# TFG: Language Models on Question Answering problems

![GitHub License](https://img.shields.io/github/license/enriquesaou/tfg-lm-qa) ![GitHub last commit](https://img.shields.io/github/last-commit/enriquesaou/tfg-lm-qa)

Author: Enrique Saiz Oubiña  
Advisor: Manuel Antonio Sánchez-Montañés Isla  
Date: June 2024

---
* [Repositories](#Repositories)
* [Resources](#Resources)
	* [Courses](#Courses)
	* [Documentation](#Documentation)
	* [Utils](#Utils)
	* [Tokenizers](#Tokenizers)
	* [Training and hyperparameters](#Trainingandhyperparameters)
	* [Evaluation](#Evaluation)
	* [More](#More)
---
All the notebooks in this repository can be run in Google Colab session with T4 GPU.

## <a name='Repositories'></a>Repositories
- [TFG HuggingFace Collection (datasets and models)](https://huggingface.co/collections/enriquesaou/tfg-66670a768e3ed59181581e65) 
- [Github Repository (training and evaluation code)](https://github.com/enriquesaou/tfg-lm-qa)


## <a name='Resources'></a>Resources
Collection of the resources i found useful during the elaboration of my work. These are different from the main document bibliography. There is something to learn from each; i recommend to take a look.
### <a name='Courses'></a>Courses
- [HuggingFace NLP](https://huggingface.co/learn/nlp-course)
- [Large Language Model Course](https://github.com/mlabonne/llm-course)


### <a name='Documentation'></a>Documentation
- [Question Answering](https://huggingface.co/tasks/question-answering)
- [PEFT ParameterEfficient FineTuning](https://huggingface.co/docs/peft/index)
- [Intro to BitsAndBytes and Quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [BitsAndBytes: Quantization and matrix multiplication](https://huggingface.co/blog/hf-bitsandbytes-integration) 
- [Phi-2 finetuning. Adapters](https://github.com/byh711/Phi2_finetuning)
- [Generation with LLMs: strategies and pitfalls](https://huggingface.co/docs/transformers/main/llm_tutorial)

### <a name='Utils'></a>Utils
- [QA utils and scripts](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)
- [Model memory estimations](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) 
- [MRQA 2019 Shared Task](https://github.com/mrqa/MRQA-Shared-Task-2019)
- [Models Database](https://epochai.org/data/epochdb/table) 

### <a name='Tokenizers'></a>Tokenizers
- [Left padding in autoregressive models](https://ai.stackexchange.com/questions/41485/while-fine-tuning-a-decoder-only-llm-like-llama-on-chat-dataset-what-kind-of-pa)
- [Phi-2 finetuning and EOS token](https://kaitchup.substack.com/p/phi-2-a-small-model-easy-to-fine)

### <a name='Trainingandhyperparameters'></a>Training and hyperparameters
- [Performance and scalability: gradient acc, FP16, ...](https://huggingface.co/docs/transformers/v4.18.0/en/performance)
- [About training in half precision (BF16 & FP16)](https://huggingface.co/microsoft/phi-2/discussions/19#657b81a7eda715a4be3c1642)
- [Half-precision in T5](https://discuss.huggingface.co/t/training-loss-0-0-validation-loss-nan/27950/4)
- [About batch sizes (Nvidia)](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size)
- [About learning rate schedulers](https://www.reddit.com/r/MachineLearning/comments/oy3co1/d_how_to_pick_a_learning_rate_scheduler/)
- [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one)
- [Finetuning on consumer hardware (PyTorch)](https://pytorch.org/blog/finetune-llms/)
- [Large research of training parameters (StableDiffusion)](https://github.com/d8ahazard/sd_dreambooth_extension/discussions/547/)

### <a name='Evaluation'></a>Evaluation
- [Evaluation considerations](https://huggingface.co/docs/evaluate/considerations)
- [The compute_metrics function](https://stackoverflow.com/questions/75744031/why-do-we-need-to-write-a-function-to-compute-metrics-with-huggingface-questio)
- [Further evaluation: BERT Score](https://huggingface.co/spaces/evaluate-metric/bertscore)
- [Further evaluation: Semantic Answer Similarity](https://www.deepset.ai/blog/semantic-answer-similarity-to-evaluate-qa)
- [Further evaluation: Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness)

### <a name='More'></a>More
- [The illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
- [Some motivation](https://www.youtube.com/watch?v=GJDNkVDGM_s)
- [SQuAD 2 and the Null Response](https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#The-SQuAD2.0-dev-set)
- [CO2 Impact calculator](https://mlco2.github.io/impact/)

---

<p align="center">
  <img src="https://visitcount.itsvg.in/api?id=tfg-lm-qa&label=visits&color=10&icon=5&pretty=false)](https://visitcount.itsvg.in" />
</a>