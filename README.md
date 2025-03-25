# [From Words to Worth: Newborn Article Impact Prediction with LLM](https://arxiv.org/abs/2408.03934?context=cs.CL)


<p align="center">
  <img src="img\model.png" alt="icon" width="65%">
</p>

<h1 align="center">
  Using LLM as Academic Article Impact Predictor
</h1>

## 🚀 **Update Log** 
- **240808 - Eerly Access**   
  - We have released the Early Access version of our code！
- **241126 - V1.0**  We’re thrilled to announce the end of Early Access and the official release of V1.0! ✨
  - The codebase is now more organized and easier to navigate! 🧹  
  - Updated and streamlined README with detailed instructions for setup and usage. 💡
  - Decoupling the dataset, more LoRa adapters weight download links, and more! 🔄  
  - Known Issues: The functionality for building the NAID dataset has not been tested on other machines, which may lead to potential issues. We plan to replace this function with a more powerful framefowk in our [another codebase](https://github.com/ssocean/PyBiblion).
- **241204 - Huggingface Spaces Support🥰** [(Click Here for Free Online Demo)](https://huggingface.co/spaces/ssocean/Newborn_Article_Impact_Predict)
  - We've set up an online demo on Hugging Face Spaces—now you can easily give it a try without writing a single line of code!
- **241210 - The paper has now been accepted by AAAI 2025!**



[//]: # (### [Early Access Version])

[//]: # (###### This [paper]&#40;https://arxiv.org/abs/2408.03934?context=cs.CL&#41; is currently under peer review. The code might change frequently. We are currently experiencing a severe staff shortage. If you encounter any issues during the replication process, please feel free to contact us through an issue or via email：oceanytech@gmail.com.)

<!-- If you have any issues, feel free to reach out via Email: oceanytech@gmail.com or open an issue in the repository. -->

## Introduction

This repository contains the official implementation for the paper [**"From Words to Worth: Newborn Article Impact Prediction with LLM"**](https://sway.cloud.microsoft/KOH09sPR21Ubojbc). The tool is designed to PEFT the LLMs for the prediction of the future impact.

## Quick Try (for most researchers)
First, pull the repo and type following commands in the console:
```
cd ScImpactPredict
pip install -r requirements.txt
```

To begin with default setting, you should request access and download the LLaMA-3 pretrain [weights](https://huggingface.co/abacusai/Llama-3-Smaug-8B) at huggingface official sites.
Then, download the provided LLaMA-3 LoRA weights (runs_dir) [here](https://drive.google.com/file/d/1YLtKjgATqAs4rApG2UzvZLRpvDGS4sse/view?usp=sharing).

After that, modify the path to the model's weights in the `demo.py` file, and type `python demo.py` in the console.



## Fine-tuning (to reproduce, optional)

For fine-tuning, you may manually modify the 'xxxForSequenceClassification' in the `transformers` package (see llama_for_naip/NAIP_LLaMA.py for more details). Or follow the [instruction](https://huggingface.co/docs/transformers/v4.27.1/en/custom_models#using-a-model-with-custom-code) to use custom code.

Then, prepare `train.sh` bash file like below:
```
DATA_PATH="ScImpactPredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="ScImpactPredict/NAID/NAID_test_extrainfo.csv"

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir official_runs/LLAMA3 \
    --checkpoint  path_to_huggingface_LLaMA3
```
Finally, type `sh train.sh` in the console. Wating for the training ends~

## Testing (to reproduce, optional)
Similar to fine-tuning, prepare `test.sh` as below:
```
python official_test.py \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --weight_dir path_to_runs_dir
```
Then, type `sh test.sh`.


## Model Weights

We also offer the weights of other models for download.

| LLMs    | Size | MAE   | NDCG  | Mem    | Download Link                                                                                  |
| ------- | ---- | ----- | ----- | ------ | ---------------------------------------------------------------------------------------------- |
| Phi-3   | 3.8B | 0.226 | 0.742 | 6.2GB  | [Download](https://drive.google.com/file/d/1OtZx8L6nyvLav4KYacvfGdG40pCPhn9a/view?usp=sharing) |
| Falcon  | 7B   | 0.231 | 0.740 | 8.9GB  | [Download](https://drive.google.com/file/d/18JGDvHLXDpsQyawIEVvJ_08HhBs-boMt/view?usp=sharing) |
| Qwen-2  | 7B   | 0.223 | 0.774 | 12.6GB | [Download](https://drive.google.com/file/d/1kq9xckxGqjJAnhtLla--vs_0yozJcvI4/view?usp=sharing) |
| Mistral | 7B   | 0.220 | 0.850 | 15.4GB | [Download](https://drive.google.com/file/d/1Rgx-_yLfXt7jTVEmdql6xSZk8vhzmBCV/view?usp=sharing) |
| Llama-3 | 8B   | 0.216 | 0.901 | 9.4GB  | [Download](https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing) |

## Compare with Previous Methods 
With a few adjustments based on your specific needs, it should work fine. Since these models train very quickly (less than a few minutes on a single RTX 3080), we won’t be providing the trained weights.

##### Repo Structure Description
Folders like retriever, database, and tools are used for building the NAID and TKPD datasets. They have no direct connection to training or inference.

### We are pretty confident in our methodology and experiments, and you should be able to achieve any of the performance reported in our paper within an acceptable margin.

## BibTex
```
@article{Zhao2024NAIP,
  title={From Words to Worth: Newborn Article Impact Prediction with LLM},
  author={Penghai Zhao and Qinghua Xing and Kairan Dou and Jinyu Tian and Ying Tai and Jian Yang and Ming-Ming Cheng and Xiang Li},
  journal={ArXiv},
  year={2024},
  volume={abs/2408.03934},
  url={https://api.semanticscholar.org/CorpusID:271744831}
}
```
