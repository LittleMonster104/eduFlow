<p align="center" width="50%">
<img src="assets/logo.png" alt="LMFlow" style="width: 50%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

# LMFlow

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/OptimalScale/LMFlow/blob/main/readme/README_es.md">Español</a> |
     
    <p>
</h4>

<p align="center" width="100%">
<img src="assets/features.png" alt="LMFlow-features" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>




## Table of Contents
- [LMFlow](#lmflow)
  - [Latest News](#latest-news)
  - [Table of Contents](#table-of-contents)
  - [Supported Models](#supported-models)
  - [Quick Start](#quick-start)
    - [Setup](#setup)
    - [Prepare Dataset](#prepare-dataset)
    - [Finetuning](#finetuning)
      - [Full Finetuning](#full-finetuning)
      - [LISA](#lisa)
      - [LoRA](#lora)
    - [Inference](#inference)
    - [Deployment](#deployment)
    - [Evaluation](#evaluation)
  - [Supported Features](#supported-features)
  - [Support](#support)
  - [License](#license)
  - [Citation](#citation)

## Supported Models

|  Model  | Conversation Template (Details) |
|  :---:  | :-------------------: |
| [Hymba](https://huggingface.co/collections/nvidia/hymba-673c35516c12c4b98b5e845f) | `Hymba` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#hymba)) |
| [DeepSeek](https://huggingface.co/deepseek-ai) | `deepseek` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#deepseek)) |
| [Gemma](https://huggingface.co/google) | `gemma` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#gemma)) |
| [InternLM2](https://huggingface.co/internlm) | `internlm2` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#internlm2)) |
| [LLaMA-2](https://huggingface.co/meta-llama) | `llama2` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#llama-2)) |
| [LLaMA-3](https://huggingface.co/meta-llama) | `llama3` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#llama-3)) |
| [Phi-3](https://huggingface.co/microsoft) | `phi3` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#phi-3)) |
| [Qwen1.5 <br> Qwen2](https://huggingface.co/Qwen) | `qwen2` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#qwen-2)) |
| [Yi](https://huggingface.co/01-ai) | `chatml` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#yi)) |
| [Yi-1.5](https://huggingface.co/01-ai) | `yi1_5` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#yi-15)) |
| [Zephyr](https://huggingface.co/HuggingFaceH4) | `zephyr` ([Link](https://optimalscale.github.io/LMFlow/examples/supported_conversation_template.html#zephyr)) |


## Quick Start

### Setup

Our package has been tested on Linux OS (Ubuntu 20.04). Other OS platforms (MacOS, Windows) are not fully tested, where you may encounter unexpected errors. If you are using LMFlow for the first time, we recommend you to try on a Linux machine or Google Colab.

CUDA versions 10.3-11.7 are supported in versions `v0.0.5` or older. For CUDA versions greater than 11.7, one can use our stable branch `>= v0.0.6`.

```bash
git clone -b v0.0.9 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .
```

> [!TIP]
> We use WandB to track and visualize the training process by default. Before running the training scripts, users may need to log in to WandB using the command: 
>```bash
>wandb login
>```
> For detailed instructions, refer to the [WandB Quickstart Guide](https://docs.wandb.ai/quickstart/). Step 1 (registration) and Step 2 (login using your WandB API key) should be sufficient to set up your environment.
>
> <details><summary>Disabling wandb</summary>  
>
> One can disable wandb by either:  
>
> 1. Adding environment variable before running the training command.
>
>```bash
>export WANDB_MODE=disabled
>```
>
> 2. OR, specifying the integrations to report the results and logs to. In the training script, add:
>
>```bash
>--report_to none \
>```
>
> </details>

### Prepare Dataset

Please refer to our [doc](https://optimalscale.github.io/LMFlow/examples/DATASETS.html).

### Finetuning

#### Full Finetuning

Full training updates all the parameters to finetune a language model.
Here is an example to finetune a GPT-2 base model.

```sh
cd data && ./download.sh alpaca && cd -

bash ./scripts/run_finetune.sh \
  --model_name_or_path gpt2 \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_gpt2
```

> [!TIP]
> For conversation dataset, specify a conversation template for better performance by adding `--conversation_template` to the command. 
> 
> <details><summary>Llama-3-8B conversation dataset example</summary>  
> 
>```bash
>cd data && ./download.sh alpaca && cd -
>
>bash ./scripts/run_finetune.sh \
>  --model_name_or_path meta-llama/Meta-Llama-3-8B \
>  --dataset_path data/alpaca/train_conversation \
>  --conversation_template llama3 \
>  --output_model_path output_models/finetuned_llama3_8b
>```
> </details>

#### LISA

[LISA](https://arxiv.org/abs/2403.17919) is a memory-efficient finetuning algorithm that allows tradeoff between memory and the number of randomly unfreezed layers. This script currently is only tested in single gpus. Please stay tuned for our latest updates :smile:
```sh
cd data && ./download.sh alpaca && cd -

bash ./scripts/run_finetune_with_lisa.sh \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_path data/alpaca/train_conversation \
  --output_model_path output_models/finetuned_llama2_7b \
  --lisa_activated_layers 1 \
  --lisa_interval_steps 20
```

> [!TIP]
> <details><summary>Llama-2-7B conversation dataset example</summary>  
> 
>```bash
>cd data && ./download.sh alpaca && cd -
>
>bash ./scripts/run_finetune_with_lisa.sh \
>  --model_name_or_path meta-llama/Llama-2-7b-hf \
>  --dataset_path data/alpaca/train_conversation \
>  --conversation_template llama2 \
>  --output_model_path output_models/finetuned_llama2_7b_lisa \
>  --lisa_activated_layers 1 \
>  --lisa_interval_steps 20
>```
> </details>

#### LoRA

LoRA is a parameter-efficient finetuning algorithm and is more efficient than full finetuning.
```sh
cd data && ./download.sh alpaca && cd -

bash ./scripts/run_finetune_with_lora.sh \
  --model_name_or_path facebook/galactica-1.3b \
  --dataset_path data/alpaca/train_conversation \
  --output_lora_path output_models/finetuned_galactica_lora
```

> [!TIP]
> <details><summary>Llama-2-7B conversation dataset example</summary>  
> 
>```bash
>cd data && ./download.sh alpaca && cd -
>
>bash ./scripts/run_finetune_with_lora.sh \
>  --model_name_or_path meta-llama/Llama-2-7b-hf \
>  --dataset_path data/alpaca/train_conversation \
>  --conversation_template llama2 \
>  --output_model_path output_models/finetuned_llama2_7b_lora \
>```
> </details>
>
> <details><summary>Merge LoRA Weight</summary>
>
>Merge LoRA weight and the base model into one using:  
>```sh
>bash ./scripts/run_merge_lora.sh \
>  --model_name_or_path Qwen/Qwen1.5-1.8B \
>  --lora_model_path output_models/lora \
>  --output_model_path output_models/lora_merged \
>```
></details>

### Inference
After finetuning, you can run the following command to chat with the model.
```sh
bash ./scripts/run_chatbot.sh output_models/finetuned_gpt2
```

> [!TIP]
> We recommend using vLLM for faster inference.
> 
> <details><summary>Faster inference using vLLM</summary>  
>
>```bash
>bash ./scripts/run_vllm_inference.sh \
>   --model_name_or_path Qwen/Qwen2-0.5B \
>   --dataset_path data/alpaca/test_conversation \
>   --output_dir data/inference_results \
>```
> </details>

### Deployment
If you want to deploy your own model locally, we provide a gradio-based UI for building chatbots. 
Running the following command will launch the demo for robin-7b:

```sh
pip install gradio
python ./examples/chatbot_gradio.py --deepspeed configs/ds_config_chatbot.json --model_name_or_path YOUR-LLAMA  --lora_model_path ./robin-7b --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: {input_text}###Assistant:"       --end_string "#" --max_new_tokens 200
```


### Evaluation
[LMFlow Benchmark](https://blog.gopenai.com/lmflow-benchmark-an-automatic-evaluation-framework-for-open-source-llms-ef5c6f142418) is an automatic evaluation framework for open-source large language models.
We use negative log likelihood (NLL) as the metric to evaluate different aspects of a language model: chitchat, commonsense reasoning, and instruction following abilities.

You can directly run the LMFlow benchmark evaluation to obtain the results to participate in the
[LLM comparision](https://docs.google.com/spreadsheets/d/1JYh4_pxNzmNA9I0YM2epgRA7VXBIeIGS64gPJBg5NHA/edit?usp=sharing).
For example, to run GPT2 XL, one may execute
```sh
bash ./scripts/run_benchmark.sh --model_name_or_path gpt2-xl
```
`--model_name_or_path` is required, you may fill in huggingface model name or local model path here.

To check the evaluation results, you may check `benchmark.log` in `./output_dir/gpt2-xl_lmflow_chat_nll_eval`,
`./output_dir/gpt2-xl_all_nll_eval` and `./output_dir/gpt2-xl_commonsense_qa_eval`.

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
