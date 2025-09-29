

<p align="center">
  <img src="/img/naip_hr.png" alt="NAIP Framework Overview" width="30%">
</p>

# Framework for newborn article impact & quality estimation.

## Overview [![Hugging Face Spaces](https://img.shields.io/badge/%20Try%20Free%20Demo-orange?logo=huggingface)](https://huggingface.co/spaces/ssocean/Newborn_Article_Impact_Predict)


The NAIP series uses fine-tuned LLMs to quickly predict the **impact** or **quality** of articles based on their internal content. 



| Version | Input              | Output                  | Model Weights                                                                 | Homepage                                                                 | Paper                                         |
|---------|--------------------|-------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------------|
| v1      | Title & Abstract   | Impact Estimation (0–1) | [Link](https://huggingface.co/ssocean/NAIPv1)                                 | [Link](https://sway.cloud.microsoft/KOH09sPR21Ubojbc)                    | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32106/34261) |
| v2      | Title & Abstract   | Quality Estimation      | [Link](https://huggingface.co/ssocean/NAIPv2)                                 | [Link](https://sway.cloud.microsoft/Pr42npP80MfPhvj8)                    | [In Preparation](#)                           |

[//]: # (This repository contains the official implementation for the paper [**"From Words to Worth: Newborn Article Impact Prediction with LLM"**]&#40;https://sway.cloud.microsoft/KOH09sPR21Ubojbc&#41;. The tool is designed to PEFT the LLMs for the prediction of the future impact.)



## 🚀 **Update Log**
- **250930 – Introducing NAIPv2: extending the series with an emphasis on quality estimation.**
- **241210 - The paper has now been accepted by AAAI 2025!**
- **241204 - Huggingface Spaces Support🥰** 
  - We've set up an online demo on Hugging Face Spaces—now you can easily give it a try without writing a single line of code!
- **241126 - V1.0**  We’re thrilled to announce the end of Early Access and the official release of V1.0! ✨
  - The codebase is now more organized and easier to navigate! 🧹  
  - Updated and streamlined README with detailed instructions for setup and usage. 💡
  - Decoupling the dataset, more LoRa adapters weight download links, and more! 🔄  
  - Known Issues: The functionality for building the NAID dataset has not been tested on other machines, which may lead to potential issues. We plan to replace this function with a more powerful framefowk in our [another codebase](https://github.com/ssocean/PyBiblion).
- **240808 - Eerly Access**   
  - We have released the Early Access version of our code！


## Quick Deployment (for most researchers)
First, pull the repo and type following commands in the console:
```
git clone https://github.com/ssocean/NAIP.git
cd NAIP
pip install -r requirements.txt
```
- To try **v1**, please use `demo_v1.py`.  
- To try **v2**, please use `demo_v2.py`.  
- You may need to download the corresponding model weights.  
- When providing the **title** and **abstract**, please avoid line breaks, LaTeX symbols, or other special formatting.  

## Reproducing NAIPv1 (optional) 
##### The following instructions are outdated. We are undergoing a major code refactoring. An updated version will be released after 2025.10.7.
### Fine-tuning
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

### Testing
Similar to fine-tuning, prepare `test.sh` as below:
```
python official_test.py \
 --data_path NAIP/NAID/NAID_test_extrainfo.csv \
 --weight_dir path_to_runs_dir
```
Then, type `sh test.sh`.


## Reproducing NAIPv2 (optional)
##### Preliminary code and dataset are released at ./v2_resource, detailed instructions will be released after 2025.10.7. 🚀 (Core team members are on vacation 🏖️)


## 🛠️ Technical Support
If you would like to conduct **comparison experiments** with NAIP but encounter difficulties in setting up the environment or reproducing the code, we provide **free technical support**.

Simply send us a `.csv` file containing the **"title"** and **"abstract"** fields, and we will return the prediction results to you.  
- In urgent cases, results can be provided **within one day**.  
- This service is free of charge and intended to facilitate **fair, reproducible comparisons** in research.  

📩 Please contact us via [oceanytech@gmail.com].


## 📚 Citation
If you find this work useful, please cite:

```bibtex
@article{Zhao2024NAIP,
  title={From Words to Worth: Newborn Article Impact Prediction with LLM},
  author={Penghai Zhao and Qinghua Xing and Kairan Dou and Jinyu Tian and Ying Tai and Jian Yang and Ming-Ming Cheng and Xiang Li},
  journal={ArXiv},
  year={2024},
  volume={abs/2408.03934},
  url={https://api.semanticscholar.org/CorpusID:271744831}
}
```



[//]: # ()
[//]: # (## Model Weights)

[//]: # ()
[//]: # (We also offer the weights of other models for download.)

[//]: # ()
[//]: # (| LLMs    | Size | MAE   | NDCG  | Mem    | Download Link                                                                                  |)

[//]: # (| ------- | ---- | ----- | ----- | ------ | ---------------------------------------------------------------------------------------------- |)

[//]: # (| Phi-3   | 3.8B | 0.226 | 0.742 | 6.2GB  | [Download]&#40;https://drive.google.com/file/d/1OtZx8L6nyvLav4KYacvfGdG40pCPhn9a/view?usp=sharing&#41; |)

[//]: # (| Falcon  | 7B   | 0.231 | 0.740 | 8.9GB  | [Download]&#40;https://drive.google.com/file/d/18JGDvHLXDpsQyawIEVvJ_08HhBs-boMt/view?usp=sharing&#41; |)

[//]: # (| Qwen-2  | 7B   | 0.223 | 0.774 | 12.6GB | [Download]&#40;https://drive.google.com/file/d/1kq9xckxGqjJAnhtLla--vs_0yozJcvI4/view?usp=sharing&#41; |)

[//]: # (| Mistral | 7B   | 0.220 | 0.850 | 15.4GB | [Download]&#40;https://drive.google.com/file/d/1Rgx-_yLfXt7jTVEmdql6xSZk8vhzmBCV/view?usp=sharing&#41; |)

[//]: # (| Llama-3 | 8B   | 0.216 | 0.901 | 9.4GB  | [Download]&#40;https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing&#41; |)
[//]: # ()
[//]: # (## Compare with Previous Methods )

[//]: # (With a few adjustments based on your specific needs, it should work fine. Since these models train very quickly &#40;less than a few minutes on a single RTX 3080&#41;, we won’t be providing the trained weights.)

[//]: # ()
[//]: # (##### Repo Structure Description)

[//]: # (Folders like retriever, database, and tools are used for building the NAID and TKPD datasets. They have no direct connection to training or inference.)
