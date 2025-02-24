# Very Useful Repo about LLM

现在的深度学习大模型有很多现成可用的工具要学会利用已有的工具，这样能提高效率，也方便能从别人的代码里学习。
## [Sentence Transformers](https://www.sbert.net/)

可以很方便的获取embedding，集成了很多MTEB上面优秀的模型，方便训练微调。

> Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art text and image embedding models. It can be used to compute embeddings using Sentence Transformer models ([quickstart](https://www.sbert.net/docs/quickstart.html#sentence-transformer)) or to calculate similarity scores using Cross-Encoder models ([quickstart](https://www.sbert.net/docs/quickstart.html#cross-encoder)). This unlocks a wide range of applications, including [semantic search](https://www.sbert.net/examples/applications/semantic-search/README.html), [semantic textual similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html), and [paraphrase mining](https://www.sbert.net/examples/applications/paraphrase-mining/README.html).

> A wide selection of over [5,000 pre-trained Sentence Transformers models](https://huggingface.co/models?library=sentence-transformers) are available for immediate use on 🤗 Hugging Face, including many of the state-of-the-art models from the [Massive Text Embeddings Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Additionally, it is easy to [train or finetune your own models](https://www.sbert.net/docs/sentence_transformer/training_overview.html) using Sentence Transformers, enabling you to create custom models for your specific use cases.
> ![[Pasted image 20250224173824.png]]


## [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master)

实现了BGE等BAAI自己提出的一系列模型，也集成了Finetune功能，但是框架只能在自己的模型上用，通用型有限，不过也可以当做代码参考。

> BGE (BAAI General Embedding) focuses on retrieval-augmented LLMs, consisting of the following projects currently:
> ![[Pasted image 20250224165701.png]]



## [LLama Factory](https://github.com/hiyouga/LLaMA-Factory)

上面集成有非常多的模型和数据和微调方法，微调不二之选，但是训练数据局限于QA和Preference类型的数据，任务类型会比较局限。

> Easily fine-tune 100+ large language models with zero-code [CLI](https://github.com/hiyouga/LLaMA-Factory#quickstart) and [Web UI](https://github.com/hiyouga/LLaMA-Factory#fine-tuning-with-llama-board-gui-powered-by-gradio)
> 
> ![[Pasted image 20250224172354.png]]
> 
> Features
> 
> - **Various models**: LLaMA, LLaVA, Mistral, Mixtral-MoE, Qwen, Qwen2-VL, DeepSeek, Yi, Gemma, ChatGLM, Phi, etc.
> - **Integrated methods**: (Continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO, etc.
> - **Scalable resources**: 16-bit full-tuning, freeze-tuning, LoRA and 2/3/4/5/6/8-bit QLoRA via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ.
> - **Advanced algorithms**: [GaLore](https://github.com/jiaweizzhao/GaLore), [BAdam](https://github.com/Ledzy/BAdam), [APOLLO](https://github.com/zhuhanqing/APOLLO), [Adam-mini](https://github.com/zyushun/Adam-mini), DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoRA+, LoftQ and PiSSA.
> - **Practical tricks**: [FlashAttention-2](https://github.com/Dao-AILab/flash-attention), [Unsloth](https://github.com/unslothai/unsloth), [Liger Kernel](https://github.com/linkedin/Liger-Kernel), RoPE scaling, NEFTune and rsLoRA.
> - **Wide tasks**: Multi-turn dialogue, tool using, image understanding, visual grounding, video recognition, audio understanding, etc.
> - **Experiment monitors**: LlamaBoard, TensorBoard, Wandb, MLflow, SwanLab, etc.
> - **Faster inference**: OpenAI-style API, Gradio UI and CLI with vLLM worker.



## [开源大模型食用指南-self-llm](https://github.com/datawhalechina/self-llm/tree/master)

大模型使用训练的中文指南，贡献者众多，有些比较详细，可能由不同贡献者决定不同风格

> 本项目是一个围绕开源大模型、针对国内初学者、基于 Linux 平台的中国宝宝专属大模型教程，针对各类开源大模型提供包括环境配置、本地部署、高效微调等技能在内的全流程指导，简化开源大模型的部署、使用和应用流程，让更多的普通学生、研究者更好地使用开源大模型，帮助开源、自由的大模型更快融入到普通学习者的生活中。

> 本项目的主要内容包括：
> 
> 1. 基于 Linux 平台的开源 LLM 环境配置指南，针对不同模型要求提供不同的详细环境配置步骤
> 2. 针对国内外主流开源 LLM 的部署使用教程，包括 LLaMA、ChatGLM、InternLM 等；
> 3. 开源 LLM 的部署应用指导，包括命令行调用、在线 Demo 部署、LangChain 框架集成等；
> 4. 开源 LLM 的全量微调、高效微调方法，包括分布式全量微调、LoRA、ptuning 等。

## [Phi Cookbook: Hands-On Examples with Microsoft's Phi Models](https://github.com/microsoft/Phi-3CookBook#phi-cookbook-hands-on-examples-with-microsofts-phi-models)

Microsoft的Phi的使用指南，有很多指南可以参考

> Phi, is a family of open AI models developed by Microsoft. Phi models are the most capable and cost-effective small language models (SLMs) available, outperforming models of the same size and next size up across a variety of language, reasoning, coding, and math benchmarks. The Phi-3 Family includes mini, small, medium and vision versions, trained based on different parameter amounts to serve various application scenarios. For more detailed information about Microsoft's Phi family, please visit the [Welcome to the Phi Family](https://github.com/microsoft/Phi-3CookBook/blob/main/md/01.Introduce/Phi3Family.md) page.
> ![[Pasted image 20250224173744.png]]