# VirusT5: Leveraging VirusT5 for Virus Host Susceptibility Prediction   
[Original VirusT5 Paper](https://github.com/vrmarathe/VirusT5)
## Overview 
VirusT5 is a transformer-based language model built on the T5 architecture, designed to predict SARS-CoV-2 evolution through a mutation-as-translation paradigm. By modeling viral mutations as a sequence-to-sequence task, VirusT5 learns to capture complex mutation patterns in the Receptor-Binding Domain (RBD) of the spike protein, identify mutation hotspots, and forecast future viral strains.

Beyond SARS-CoV-2, VirusT5 extends to diverse viral genera, demonstrating its adaptability for host classification tasks and other virus-related applications. This paper explores its effectiveness in predicting viral hosts, showcasing its potential for broader genomic analysis.

## Model Availability
The model is available to use through Huggingface [VirusT5](https://huggingface.co/vrmarathe/VirusT5)
## How To Use The Pretrained Model
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer for the VirusT5 model
tokenizer = AutoTokenizer.from_pretrained("vrmarathe/VirusT5", trust_remote_code=True)

# Load the pre-trained VirusT5 model (T5-based)
model = AutoModelForSeq2SeqLM.from_pretrained("vrmarathe/VirusT5", trust_remote_code=True,from_flax=True)
```
## Installation  
Clone the repository and set up the required dependencies:  
```bash  
git clone https://github.com/vrmarathe/VirusT5.git
cd VirusT5
cd environment
conda env create -f flax2_environment.yml
```

### Pretraining  
VirusT5 was pretrained on a large corpus of SARS-CoV-2 genome sequences to learn the underlying syntax and grammar of genomic data.  
- **Dataset**: Genome Dataset comprising 100,000 SARS-CoV-2 genome sequences from GISAID.  
- **Objective**: Masked Language Modeling (MLM) with 15% token masking using sentinel tokens.  
- **Sequence Length**: Segmented into sequences of up to 512 base pairs.  
- **Optimization**:  
  - Inverse square root learning rate schedule.  
  - Initial learning rate: 0.005 for 2,000 steps, followed by exponential decay.  
- **Training Hardware**:  
  - NDSU CCAST HPC clusters with 32 CPU cores, 100 GB RAM, and two NVIDIA A40 GPUs (40 GB each).  
- **Duration**: Pretrained for 12,000 steps.
- The scripts for the pretraining can be found in the pretraining folder  


