<p align="center" width="50%">
<img src="assets/logo.png" alt="eduFlow" style="width: 50%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

# eduFlow

<h4 align="center">
    <p>
	@@ -17,234 +17,45 @@
<img src="assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## introduction
Eduflow is a simpler and more powerful Learning Management System (LMS) for managing social learning at scale. It streamlines course creation, fosters collaboration, and provides valuable insights through data analytics for corporate training, higher education, and a variety of use cases such as onboarding and sales training. What makes Eduflow unique is its powerful social learning features and easy-to-use interface.Eduflow has revolutionized the way large-scale social learning is managed. It streamlines the course creation process, fosters interaction and collaboration among learners, and helps improve the learning experience with data-driven insights.

## Table of Contents
- [eduFlow](#eduflow)


## Supported Models

|  Model  | Conversation Template (Details) |
|  :---:  | :-------------------: |



## Quick Start


### Setup

```bash
git clone -b v0.0.9 https://github.com/LittleMonster104/eduflow.git
pip install -e .
```


### Prepare Dataset


### Finetuning


#### Full Finetuning


### Inference


### Evaluation


## Supported Features

<details> <summary>Finetune Acceleration & Memory Optimization</summary>
* LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning
  
  LISA is a novel and memory-efficient training strategy for large language models that outperforms existing methods like LoRA by selectively freezing layers during optimization. Check out [LISA](https://arxiv.org/abs/2403.17919) for more details.  
  In LMFLow, activate LISA using `--use_lisa 1` in your training command. Control the number of activation layers with `--lisa_activated_layers 2`, and adjust the freezing layers interval using `--lisa_step_interval 20`. 
* LoRA
  
  LoRA is a parameter-efficient finetuning algorithm and is more efficient than full finetuning. Check out [finetuning-lora](#finetuning-lora) for more details.
* FlashAttention
  LMFlow supports both FlashAttention-1 and the latest FlashAttention-2. Check out [flash_attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md) for more details.
* Gradient Checkpointing
  
  [Gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing) is a memory optimization technique that trades compute for memory. 
  It is useful when the model is too large to fit into GPU memory. 
  Use it by just adding `--gradient_checkpointing` to your training command.
* Deepspeed Zero3
  
  LMFlow supports [Deepspeed Zero-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html). 
  We provide an example [deepspeed config](https://github.com/OptimalScale/LMFlow/blob/main/configs/ds_config_zero3.json), and you can directly use it.
</details>
<details> <summary>Inference Acceleration</summary>
* LLaMA Inference on CPU
  Thanks to the great efforts of [llama.cpp](https://github.com/ggerganov/llama.cpp). It is possible for everyone to run their LLaMA models on CPU by 4-bit quantization. We provide a script to convert LLaMA LoRA weights to `.pt` files. You only need to use `convert-pth-to-ggml.py` in llama.cpp to perform quantization.
* FlashAttention
  LMFlow supports both FlashAttention-1 and the latest FlashAttention-2. Check out [flash_attention](https://github.com/OptimalScale/LMFlow/blob/main/readme/flash_attn2.md) for more details.
* vLLM
  Try vLLM for fast and easy-to-use LLM inference and serving. Thanks for the [great work](https://github.com/vllm-project/vllm)!
</details>
<details> <summary>Long Context</summary>
* Position Interpolation for LLaMA Models
  Now LMFlow supports the latest Linear & NTK (Neural Kernel theory) scaling techniques for LLaMA models. Check out [postion_interpolation](https://github.com/OptimalScale/LMFlow/blob/main/readme/Position_Interpolation.md) for more details.
</details>
<details> <summary>Model Customization</summary>
* Vocabulary Extension
  Now you can train your own sentencepiece tokenizer and merge it with model's origin hf tokenizer. Check out [vocab_extension](https://github.com/OptimalScale/LMFlow/blob/main/scripts/vocab_extension) for more details.
</details>
<details> <summary>Multimodal</summary>
* Multimodal Chatbot
  LMFlow supports multimodal inputs of images and texts. Check out our [LMFlow multimodal chatbot](https://github.com/OptimalScale/LMFlow/blob/main/scripts/run_vis_chatbot_gradio_minigpt4.sh).
</details>
<details> <summary>Custom Optimization</summary>
* Custom Optimization
  LMFlow now supports custom optimizer training with a variety of optimizers. Elevate your model's performance with tailored optimization strategies. Dive into the details and try out the new features with our updated script at [custom_optimizers](https://github.com/OptimalScale/LMFlow/blob/main/scripts/run_finetune_with_custom_optim.sh).
  The following table evaluates the performance of custom optimizers in the fine-tuning process of GPT-2 on the Alpaca dataset, emphasizing their individual impacts on the training loss. The specific hyperparameter settings utilize default configurations, which can be customized and adjusted at [custom_optimizers](https://github.com/OptimalScale/LMFlow/blob/main/scripts/run_finetune_with_custom_optim.sh). It is important to note that the evaluations were conducted over a duration of 0.1 epochs to provide a preliminary insight into the optimizers' effectiveness.
  | Optimizer Name | Train Loss |
  |----------------|------------|
  | RMSprop        | 2.4016     |
  | LION-32bit     | 2.4041     |
  | Adam           | 2.4292     |
  | AdamP          | 2.4295     |
  | AdamW          | 2.4469     |
  | AdaFactor      | 2.4543     |
  | AdaBound       | 2.4547     |
  | AdamWScheduleFree       | 2.4677     |
  | Adan           | 2.5063     |
  | NAdam          | 2.5569     |
  | AdaBelief      | 2.5857     |
  | AdaMax         | 2.5924     |
  | RAdam          | 2.6104     |
  | AdaDelta       | 2.6298     |
  | AdaGrad        | 2.8657     |
  | Yogi           | 2.9314     |
  | NovoGrad       | 3.1071     |
  | Sophia         | 3.1517     |
  | LAMB           | 3.2350     |
  | LARS           | 3.3329     |
  | SGDScheduleFree        | 3.3541     |
  | SGDP           | 3.3567     |
  | SGD            | 3.3734     |
</details>
## Support
If you need any help, please submit a Github issue.
## License
The code included in this project is licensed under the [Apache 2.0 license](https://github.com/OptimalScale/LMFlow/blob/main/LICENSE).
If you wish to use the codes and models included in this project for commercial purposes, please sign this [document](https://docs.google.com/forms/d/e/1FAIpQLSfJYcci6cbgpIvx_Fh1xDL6pNkzsjGDH1QIcm4cYk88K2tqkw/viewform?usp=pp_url) to obtain authorization.
## Citation
If you find this repository useful, please consider giving ⭐ and citing our [paper](https://arxiv.org/abs/2306.12420):
```
@article{diao2023lmflow,
  title={Lmflow: An extensible toolkit for finetuning and inference of large foundation models},
  author={Diao, Shizhe and Pan, Rui and Dong, Hanze and Shum, Ka Shun and Zhang, Jipeng and Xiong, Wei and Zhang, Tong},
  journal={arXiv preprint arXiv:2306.12420},
  year={2023}
}
```
```
@article{dong2023raft,
  title={Raft: Reward ranked finetuning for generative foundation model alignment},
  author={Dong, Hanze and Xiong, Wei and Goyal, Deepanshu and Pan, Rui and Diao, Shizhe and Zhang, Jipeng and Shum, Kashun and Zhang, Tong},
  journal={arXiv preprint arXiv:2304.06767},
  year={2023}
}
```
```
@article{pan2024lisa,
  title={LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning}, 
  author={Pan, Rui and Liu, Xiang and Diao, Shizhe and Pi, Renjie and Zhang, Jipeng and Han, Chi and Zhang, Tong},
  journal={arXiv preprint arXiv:2403.17919},
  year={2024}
}
```
