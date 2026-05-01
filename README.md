# 🏥 Medical Q&A Chat Assistant — CISC 886 Cloud Computing

> **Queen's University · School of Computing · Kingston, Canada**  
> An end-to-end cloud-native medical question-answering system: from raw data on S3 → PySpark preprocessing on EMR → LoRA fine-tuning on Colab → GGUF deployment on EC2 → live chat via OpenWebUI.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Repository Structure](#-repository-structure)
- [Phase 1 — AWS VPC & Networking](#-phase-1--aws-vpc--networking)
- [Phase 2 — Dataset & S3 Setup](#-phase-2--dataset--s3-setup)
- [Phase 3 — Data Preprocessing on EMR](#-phase-3--data-preprocessing-on-emr)
- [Phase 4 — Model Fine-Tuning (Colab)](#-phase-4--model-fine-tuning-colab)
- [Phase 5 — Model Deployment on EC2](#-phase-5--model-deployment-on-ec2)
- [Phase 6 — Web Interface (OpenWebUI)](#-phase-6--web-interface-openwebui)
- [Hyperparameter Table](#-hyperparameter-table)
- [AWS Cost Summary](#-aws-cost-summary)
- [Replication Checklist](#-replication-checklist)

---

## 🔭 Project Overview

This project builds a **domain-specific medical chat assistant** fine-tuned on the [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset) from Kaggle. The system is deployed end-to-end on AWS and accessible through a browser-based chat interface.

| Component | Technology |
|-----------|-----------|
| Dataset | Kaggle — Comprehensive Medical Q&A (train.csv) |
| Preprocessing | Apache PySpark on AWS EMR |
| Storage | Amazon S3 |
| Base Model | `Qwen2.5-1.5B-Instruct` (1.5 B parameters, 4-bit quantized) |
| Fine-Tuning | Unsloth + LoRA (PEFT) on Google Colab |
| Export Format | GGUF (Q4_K_M quantization) |
| Model Serving | Ollama on AWS EC2 |
| Chat Interface | OpenWebUI |
| Networking | Custom AWS VPC (no default VPC used) |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS VPC (25cdkg-vpc)                 │
│   CIDR: 10.0.0.0/16                                        │
│                                                             │
│  ┌──────────────────────┐    ┌────────────────────────┐    │
│  │  Public Subnet        │    │  EMR Cluster           │    │
│  │  10.0.1.0/24         │    │  (25cdkg-emr-cluster)  │    │
│  │                      │    │  m5.xlarge × 3 nodes   │    │
│  │  ┌────────────────┐  │    │                        │    │
│  │  │  EC2 Instance  │  │    │  PySpark Pipeline      │    │
│  │  │ (25cdkg-ec2)   │  │    │  reads  S3 raw data    │    │
│  │  │  g4dn.xlarge   │  │    │  writes S3 processed   │    │
│  │  │                │  │    └────────────┬───────────┘    │
│  │  │  Ollama :11434 │  │                 │                │
│  │  │  OpenWebUI:8080│  │    ┌────────────▼───────────┐    │
│  │  └────────────────┘  │    │  Amazon S3             │    │
│  │                      │    │  25cdkg-medical-qa/    │    │
│  │  Internet Gateway    │    │  ├── raw-data/         │    │
│  │  Route: 0.0.0.0/0   │    │  ├── processed-data/   │    │
│  └──────────────────────┘    │  └── eda/              │    │
│                              └────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
          │
          │  Browser (port 8080)
          ▼
     👤  End User
```

**Data Flow:**  
Raw CSV on S3 → EMR PySpark cleans, filters, and formats data into ChatML → processed JSON splits written back to S3 → downloaded locally → Google Colab fine-tunes Qwen2.5-1.5B with LoRA → model exported to GGUF → uploaded to EC2 → Ollama serves the model → OpenWebUI provides the browser chat interface → user interacts.

---

## ✅ Prerequisites

### Accounts & Access
- AWS account with access to EMR, EC2, S3, VPC
- Google account (for free Colab GPU)
- Kaggle account (for dataset download)
- Hugging Face account (optional — for model browsing)

### Local Tools
```bash
# AWS CLI
aws --version          # >= 2.x

# Python
python --version       # >= 3.10

# Git
git --version
```

### AWS IAM Permissions Required
- `AmazonEMRFullAccess`
- `AmazonEC2FullAccess`
- `AmazonS3FullAccess`
- `AmazonVPCFullAccess`

> ⚠️ **Resource Naming Policy:** Every AWS resource must be prefixed with your Queen's netID.  
> Example: `25cdkg-vpc`, `25cdkg-ec2`, `25cdkg-emr-cluster`, `25cdkg-s3-bucket`

---

## 📁 Repository Structure

```
.
├── Data_Preprocessing.py        # PySpark pipeline — runs on AWS EMR
├── Complete_Code_MQA.ipynb      # Local exploration notebook (PySpark + EDA)
├── Fine_Tuning.ipynb            # Unsloth LoRA fine-tuning notebook (Colab)
├── .gitignore
└── README.md
```

> **Note:** Model weights (`medical_qa_gguf/`, `medical_qa_model_lora/`) and AWS `.pem` keys are excluded by `.gitignore` and must never be committed.

---

## 🌐 Phase 1 — AWS VPC & Networking

### 1.1 Create the VPC

```bash
# Create VPC
aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=25cdkg-vpc}]'

# Enable DNS hostnames
aws ec2 modify-vpc-attribute \
  --vpc-id <VPC_ID> \
  --enable-dns-hostnames
```

### 1.2 Create Public Subnet

```bash
aws ec2 create-subnet \
  --vpc-id <VPC_ID> \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=25cdkg-subnet-public}]'

# Enable auto-assign public IPs
aws ec2 modify-subnet-attribute \
  --subnet-id <SUBNET_ID> \
  --map-public-ip-on-launch
```

### 1.3 Internet Gateway & Route Table

```bash
# Create and attach Internet Gateway
aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=25cdkg-igw}]'

aws ec2 attach-internet-gateway \
  --internet-gateway-id <IGW_ID> \
  --vpc-id <VPC_ID>

# Create route table and add default route
aws ec2 create-route-table \
  --vpc-id <VPC_ID> \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=25cdkg-rt-public}]'

aws ec2 create-route \
  --route-table-id <RT_ID> \
  --destination-cidr-block 0.0.0.0/0 \
  --gateway-id <IGW_ID>

aws ec2 associate-route-table \
  --route-table-id <RT_ID> \
  --subnet-id <SUBNET_ID>
```

### 1.4 Security Groups

```bash
# EC2 Security Group
aws ec2 create-security-group \
  --group-name 25cdkg-sg-ec2 \
  --description "EC2 SG for Ollama and OpenWebUI" \
  --vpc-id <VPC_ID>

# Open required ports
aws ec2 authorize-security-group-ingress --group-id <SG_ID> \
  --protocol tcp --port 22    --cidr 0.0.0.0/0     # SSH
aws ec2 authorize-security-group-ingress --group-id <SG_ID> \
  --protocol tcp --port 8080  --cidr 0.0.0.0/0     # OpenWebUI
aws ec2 authorize-security-group-ingress --group-id <SG_ID> \
  --protocol tcp --port 11434 --cidr 0.0.0.0/0     # Ollama API
```

| Port | Service | Reason |
|------|---------|--------|
| 22 | SSH | Remote access to EC2 |
| 8080 | OpenWebUI | Browser chat interface |
| 11434 | Ollama | LLM API endpoint |

---

## 📦 Phase 2 — Dataset & S3 Setup

### 2.1 Create S3 Bucket

```bash
aws s3api create-bucket \
  --bucket 25cdkg-medical-qa \
  --region us-east-1

# Create folder structure
aws s3api put-object --bucket 25cdkg-medical-qa --key 25cdkg-raw-data/
aws s3api put-object --bucket 25cdkg-medical-qa --key 25cdkg-processed-data/
aws s3api put-object --bucket 25cdkg-medical-qa --key 25cdkg-eda/
```

### 2.2 Download & Upload the Dataset

```bash
# Install kagglehub
pip install kagglehub

# Download (Python)
python3 -c "
import kagglehub
path = kagglehub.dataset_download('thedevastator/comprehensive-medical-q-a-dataset')
print('Downloaded to:', path)
"

# Upload raw CSV to S3
aws s3 cp /path/to/train.csv s3://25cdkg-medical-qa/25cdkg-raw-data/train.csv
```

**Dataset Summary:**

| Property | Value |
|----------|-------|
| Name | Comprehensive Medical Q&A Dataset |
| Source | Kaggle |
| License | CC BY 4.0 |
| Total Samples | ~16,400 rows |
| Columns | `qtype`, `Question`, `Answer` |
| Train Split | 70% |
| Validation Split | 15% |
| Test Split | 15% |

**Sample Record:**
```
qtype:    "symptoms"
Question: "what are the symptoms of diabetes?"
Answer:   "symptoms of diabetes include frequent urination, increased thirst, 
           unexplained weight loss, fatigue, blurred vision..."
```

> **Data Leakage Prevention:** Splits were created using PySpark `randomSplit([0.7, 0.15, 0.15], seed=42)` on the fully deduplicated dataset. Deduplication (`dropDuplicates`) was applied before splitting to ensure no duplicate Q&A pairs appear across splits.

---

## ⚡ Phase 3 — Data Preprocessing on EMR

### 3.1 Launch EMR Cluster

```bash
aws emr create-cluster \
  --name "25cdkg-emr-cluster" \
  --release-label emr-7.0.0 \
  --applications Name=Spark \
  --instance-type m5.xlarge \
  --instance-count 3 \
  --ec2-attributes SubnetId=<SUBNET_ID>,KeyName=25cdkg-key \
  --use-default-roles \
  --region us-east-1
```

**EMR Configuration:**

| Setting | Value |
|---------|-------|
| Release | EMR 7.0.0 |
| Application | Apache Spark |
| Master Node | m5.xlarge |
| Worker Nodes | 2 × m5.xlarge |
| Region | us-east-1 |

### 3.2 Upload and Run the PySpark Script

```bash
# Upload script to S3
aws s3 cp Data_Preprocessing.py s3://25cdkg-medical-qa/scripts/

# Submit PySpark job to EMR
aws emr add-steps \
  --cluster-id <CLUSTER_ID> \
  --steps Type=Spark,Name="MedicalQA-Preprocessing",\
ActionOnFailure=CONTINUE,\
Args=[s3://25cdkg-medical-qa/scripts/Data_Preprocessing.py]
```

### 3.3 Pipeline Steps Explained

The `Data_Preprocessing.py` script performs the following:

| Step | Description |
|------|-------------|
| **1** | Initialize Spark Session |
| **2** | Load `train.csv` from S3 |
| **3** | Lowercase & trim `Question`, `Answer`, `qtype` columns |
| **4** | Drop rows with null Question or Answer |
| **5** | Compute word counts (`q_len`, `a_len`) for EDA |
| **6** | Generate 5 EDA figures and upload to S3 |
| **7** | Clean `qtype` column (extract sub-label after `:`) |
| **8** | Outlier analysis on answer lengths |
| **9** | Filter: `10 ≤ a_len ≤ 500` and `q_len ≥ 5` |
| **10** | Deduplicate on `(Question, Answer)` |
| **11** | Format into **ChatML** template |
| **12** | Final length filter: `len(Question) > 20`, `len(Answer) > 40` |
| **13** | Split 70/15/15 and save `train/`, `validation/`, `test/` JSON to S3 |

**ChatML Format Applied:**
```
<|im_start|>system
You are a professional medical assistant. Answer the patient's questions accurately.<|im_end|>
<|im_start|>user
{Question}<|im_end|>
<|im_start|>assistant
{Answer}<|im_end|>
```

### 3.4 EDA Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Distribution of Question Word Counts |
| Fig 2 | Top 10 Question Types (Label Balance) |
| Fig 3 | Question vs Answer Length Correlation (scatter) |
| Fig 4 | Distribution of Top 5 Medical Question Types (pie) |
| Fig 5 | Sample Count per Split (Train / Validation / Test) |

### 3.5 Teardown EMR Cluster

> ⚠️ **REQUIRED:** Terminate the cluster immediately after the job completes to avoid charges.

```bash
aws emr terminate-clusters --cluster-ids <CLUSTER_ID>

# Verify terminated state
aws emr describe-cluster --cluster-id <CLUSTER_ID> \
  --query "Cluster.Status.State"
```

### 3.6 Download Processed Data Locally

```bash
# Download for fine-tuning
aws s3 cp s3://25cdkg-medical-qa/25cdkg-processed-data/train/      ./S3/train/      --recursive
aws s3 cp s3://25cdkg-medical-qa/25cdkg-processed-data/validation/  ./S3/val/        --recursive
aws s3 cp s3://25cdkg-medical-qa/25cdkg-processed-data/test/        ./S3/test/       --recursive

# Rename the part files
mv ./S3/train/part-*.json      ./S3/train.json
mv ./S3/val/part-*.json        ./S3/val.json
mv ./S3/test/part-*.json       ./S3/test.json
```

---

## 🧠 Phase 4 — Model Fine-Tuning (Colab)

> Run `Fine_Tuning.ipynb` on **Google Colab** with a T4 GPU runtime.  
> Runtime → Change runtime type → **T4 GPU**

### 4.1 Install Dependencies

```python
!pip install unsloth trl transformers datasets accelerate bitsandbytes
```

### 4.2 Model Selection

| Property | Value |
|----------|-------|
| Model | `Qwen2.5-1.5B-Instruct` |
| Source | [Unsloth Hub](https://huggingface.co/unsloth/qwen2.5-1.5b-instruct-bnb-4bit) |
| Parameters | 1.5 Billion |
| License | Apache 2.0 |
| Quantization | 4-bit (BnB) — loaded via Unsloth |
| Domain Fit | General instruction-following; adapted to medical Q&A via LoRA |
| Hardware Fit | Runs on Colab T4 (16 GB VRAM) with 4-bit quantization |

**Why Qwen2.5-1.5B?**  
At 1.5 B parameters with 4-bit quantization, the model fits comfortably in Colab's T4 GPU memory while still producing coherent, structured medical responses. Its instruction-tuned variant already understands the ChatML prompt format used in our preprocessed data.

### 4.3 Load Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/qwen2.5-1.5b-instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

### 4.4 Add LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

### 4.5 Training

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        warmup_steps=50,
        num_train_epochs=2,
        learning_rate=2e-4,
        bf16=True,                        # T4 supports bfloat16
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        eval_strategy="steps",
        eval_steps=500,
    ),
)

trainer.train()
```

### 4.6 Export to GGUF

```python
model.save_pretrained_gguf(
    "medical_qa_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

Then upload the `.gguf` file to your EC2 instance:

```bash
scp -i 25cdkg-key.pem medical_qa_gguf/model.gguf \
    ec2-user@<EC2_PUBLIC_IP>:/home/ec2-user/models/
```

---

## 🚀 Phase 5 — Model Deployment on EC2

### 5.1 Launch EC2 Instance

```bash
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \   # Amazon Linux 2023
  --instance-type g4dn.xlarge \
  --key-name 25cdkg-key \
  --security-group-ids <SG_ID> \
  --subnet-id <SUBNET_ID> \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=25cdkg-ec2}]' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50}}]'
```

**EC2 Configuration:**

| Setting | Value |
|---------|-------|
| Instance Type | `g4dn.xlarge` |
| AMI | Amazon Linux 2023 |
| vCPUs | 4 |
| RAM | 16 GB |
| GPU | NVIDIA T4 (16 GB VRAM) |
| Storage | 50 GB EBS |

### 5.2 Connect to EC2

```bash
ssh -i 25cdkg-key.pem ec2-user@<EC2_PUBLIC_IP>
```

### 5.3 Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Verify
ollama --version
```

### 5.4 Load the Fine-Tuned Model

```bash
# Create models directory
mkdir -p ~/models

# Create Modelfile
cat > ~/Modelfile << 'EOF'
FROM /home/ec2-user/models/model.gguf

SYSTEM "You are a professional medical assistant trained to answer patient questions accurately and clearly."

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
EOF

# Register model in Ollama
ollama create 25cdkg-medical-qa -f ~/Modelfile

# Verify model is loaded
ollama list
```

### 5.5 Test the Model API

```bash
# Quick terminal test
ollama run 25cdkg-medical-qa "What are the symptoms of diabetes?"

# curl API test
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "25cdkg-medical-qa",
    "prompt": "What are common treatments for hypertension?",
    "stream": false
  }'
```

---

## 🌐 Phase 6 — Web Interface (OpenWebUI)

### 6.1 Install Docker

```bash
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
```

### 6.2 Run OpenWebUI

```bash
docker run -d \
  --name 25cdkg-openwebui \
  --restart always \
  -p 8080:8080 \
  -e OLLAMA_BASE_URL=http://host-gateway:11434 \
  --add-host=host-gateway:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

> The `--restart always` flag ensures OpenWebUI starts automatically on server reboot.

### 6.3 Access the Interface

Open your browser and navigate to:

```
http://<EC2_PUBLIC_IP>:8080
```

1. Create an admin account on first launch.
2. Select **25cdkg-medical-qa** from the model dropdown.
3. Start chatting with the fine-tuned medical assistant.

### 6.4 Verify Auto-Start on Reboot

```bash
# Reboot the instance
sudo reboot

# After reconnecting, verify both services are running
sudo systemctl status ollama
docker ps | grep openwebui
```

---

## 📊 Hyperparameter Table

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen2.5-1.5B-Instruct` |
| Total Parameters | 1.5 Billion |
| Quantization (loading) | 4-bit (BitsAndBytes) |
| Fine-Tuning Method | LoRA (PEFT via Unsloth) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning Rate | 2e-4 |
| Batch Size (per device) | 1 |
| Gradient Accumulation Steps | 8 (effective batch = 8) |
| Warmup Steps | 50 |
| Number of Epochs | 2 |
| Optimizer | AdamW 8-bit |
| Weight Decay | 0.01 |
| LR Scheduler | Linear |
| Max Sequence Length | 2048 |
| Export Quantization | Q4_K_M (GGUF) |
| Training Hardware | Google Colab T4 GPU |
| Inference Hardware | AWS EC2 g4dn.xlarge |

---

## 💰 AWS Cost Summary

| Service | Configuration | Approx. Cost |
|---------|---------------|-------------|
| **EMR Cluster** | 3 × m5.xlarge, ~2 hours | ~$1.50 |
| **EC2 Instance** | g4dn.xlarge, on-demand | ~$0.526/hr |
| **S3 Storage** | < 5 GB (data + EDA figures) | ~$0.12/month |
| **Data Transfer** | S3 ↔ EMR (same region) | ~$0.00 |
| **VPC / Networking** | Internet Gateway, route tables | ~$0.00 |
| **Google Colab** | Free T4 GPU (fine-tuning) | $0.00 |

> 💡 **Cost Tip:** Terminate the EMR cluster immediately after preprocessing. For EC2, stop the instance when not actively demoing to avoid continuous charges.

---

## ✅ Replication Checklist

```
[ ] AWS CLI configured (aws configure)
[ ] S3 bucket created with correct folder structure
[ ] Dataset downloaded from Kaggle and uploaded to S3
[ ] VPC, subnet, IGW, route table, and security groups created
[ ] EMR cluster launched and PySpark job submitted
[ ] EMR cluster TERMINATED after job completion (screenshot required)
[ ] Processed JSON files downloaded from S3
[ ] Fine_Tuning.ipynb run on Colab with T4 GPU
[ ] GGUF model exported and transferred to EC2
[ ] Ollama installed and model registered on EC2
[ ] curl test to Ollama API successful
[ ] OpenWebUI running on port 8080 with auto-restart
[ ] Browser chat session working with fine-tuned model visible
```

---

## 📚 References

- [Unsloth Fine-Tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- [Qwen2.5 on Hugging Face](https://huggingface.co/unsloth/qwen2.5-1.5b-instruct-bnb-4bit)
- [Comprehensive Medical Q&A Dataset — Kaggle](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset)
- [Ollama Documentation](https://ollama.com/docs)
- [OpenWebUI GitHub](https://github.com/open-webui/open-webui)
- [AWS EMR PySpark Documentation](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark.html)

---

<p align="center">
  <b>CISC 886 — Cloud Computing · Queen's University · 2025</b>
</p>
