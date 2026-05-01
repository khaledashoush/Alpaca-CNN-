# 🏥 MedAssist — Cloud-Based Medical QA Chat Assistant

**CISC 886 — Cloud Computing | Queen's University NetID:** 25cdkg

**Student:** Salma Essam & Khaled Ashoush & Fatma Ahmed

A cloud-based conversational chatbot designed to act as a **Professional Medical Assistant**. It leverages a fine-tuned **Qwen 2.5 1.5B Instruct** LLM trained on **17,023** structured medical Q&A records derived from the [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset/data), deployed entirely on AWS infrastructure.



---

## 🚀 Quickstart (TL;DR)

```bash
Step 1 → AWS Console          # Create VPC, Subnet, SG, S3
Step 2 → aws s3 cp train.csv  # Upload raw data to S3
Step 3 → EMR Cluster (PySpark) # Preprocess 38K records → JSONL
Step 4 → GPU Machine          # LoRA fine-tune → export GGUF
Step 5 → EC2 Instance         # Ollama + OpenWebUI (Docker)
Step 6 → Teardown             # Terminate all resources
```

> Full commands for each step are in the sections below.

---

## 🏗️ System Architecture



<img width="1600" height="511" alt="image" src="https://github.com/user-attachments/assets/9a5b61f9-5610-403e-b349-9748329a70c8" />




### Data Flow

1. Raw medical Q&A dataset uploaded from Kaggle to **S3**
2. **EMR cluster** (m5.xlarge × 3) runs PySpark preprocessing in Public Subnet (10.0.2.0/24)
3. Processed JSONL files saved back to **S3** via HTTPS (port 443)
4. Fine-tuning notebook trains the model using **LoRA (PEFT)** on GPU hardware
5. Fine-tuned model exported to **GGUF format** for efficient inference
6. GGUF model loaded into **Ollama** (LLM Runner) on **EC2** in Public Subnet A (10.0.1.0/24)
7. **OpenWebUI** sends LLM requests to Ollama on the same EC2 host
8. NAT traffic routed through **Internet Gateway** (25cdkg-igw) with route `0.0.0.0/0`
9. Users access the medical assistant through their browser via **port 3000**
10. EC2 communicates with S3 via **S3 API (port 443)** for data and model downloads

---

## 📁 Repository Structure

```
.
├── unsloth_compiled_cache/       # Auto-generated cache created by Unsloth during training
├── project_instructions          # Project Requirement
├── medical_qa_gguf/              # Fine-tuned model in GGUF format — used by Ollama for inference on EC2
├── medical_qa_model_lora/        # LoRA adapter weights — the trained parameters from fine-tuning
├── S3/                           # Local copy of S3 data (train.jsonl, val.jsonl, test_qa.jsonl) 
├── README.md                     # This file
├── .gitignore                    # Excludes large model files and keys
├── Data_Preprocessing.ipynb      # PySpark preprocessing notebook (Section 4)
├── Fine_Tuning.ipynb             # Model fine-tuning notebook (Section 5)
├── Complete_Code_MQA.ipynb       # Combined complete pipeline notebook
├── install_deps.txt              # Python dependencies list
├── AWS_Images/                   # AWS Console & architecture screenshots
│   ├── architecture_diagram.png  # System architecture diagram
│   ├── emr_cluster_config.png
│   ├── emr_terminated.png
│   ├── s3_output_files.png
│   ├── vpc_config.png
│   ├── ec2_instance.png
│   ├── ollama_serving.png
│   ├── curl_response.png
│   ├── openwebui_interface.png
│   ├── openwebui_conversation.png
│   ├── cost_by_service_chart.png
│   └── cost_by_service_table.png
├── EDA_Images/                   # Exploratory Data Analysis figures
│   ├── fig1_question_length.png
│   ├── fig2_label_balance.png
│   ├── fig3_q_vs_a_scatter.png
│   ├── fig4_top5_categories.png
│   └── fig5_split_counts.png
└── Model_Images/                 # Training and evaluation figures
    ├── training_curves.png
    └── evaluation_metrics.png
```

---

## 📋 Prerequisites

| Requirement | Version / Notes |
|---|---|
| **AWS Account** | Region: **us-east-1b** |
| **Python** | 3.10+ |
| **Apache Spark** | 3.x (on EMR — no local install needed) |
| **GPU (for fine-tuning)** | ≥16 GB VRAM (local GPU or Google Colab T4 free tier) |
| **AWS CLI** | Configured with project credentials (`aws configure`) |
| **Docker** | Required on EC2 for OpenWebUI |

### Python Dependencies

```bash
# Preprocessing
pip install pyspark kagglehub matplotlib seaborn pandas numpy

# Fine-tuning (requires CUDA GPU)
pip install unsloth trl peft accelerate bitsandbytes transformers datasets

# Evaluation
pip install rouge-score nltk scikit-learn
```

Or install all at once:

```bash
pip install -r install_deps.txt
```

> **Note:** The fine-tuning notebook requires a CUDA-compatible GPU with ≥16 GB VRAM. Google Colab (T4 GPU, free tier) is sufficient as an alternative.

### IAM Requirements

| Policy | Purpose |
|---|---|
| `AmazonS3FullAccess` | Upload/download data and model artefacts |
| `AmazonEC2FullAccess` | Launch and manage EC2 instances |
| `AmazonEMRFullAccess` | Create and terminate EMR clusters |
| `IAMFullAccess` (or scoped) | Attach instance profiles to EMR/EC2 |

> EMR also requires a service role (`AmazonEMR-ServiceRole`) and an EC2 instance profile (`AmazonEMR-InstanceProfile`) — these are created automatically when you launch EMR via the console for the first time.

---

## 📊 Dataset

| Field | Details |
|---|---|
| **Name** | Comprehensive Medical Q&A Dataset |
| **Source** | Kaggle |
| **Raw Records** | ~38,000 |
| **After Filtering** | 17,023 |
| **Format** | CSV → JSONL (after preprocessing) |

### Train / Validation / Test Split

| Split | Samples | Percentage |
|---|---|---|
| **Train** | ~12,033 | 70% |
| **Validation** | ~2,507 | 15% |
| **Test** | ~2,483 | 15% |

> No overlap between train, validation, and test sets.

### EDA Figures (5 total — available in `EDA_Images/`)

| Figure | Description |
|---|---|
| **Fig 1** | Question word count distribution — most questions are 5–20 words |
| **Fig 2** | Top 10 question types (label balance) — identifies dominant medical categories |
| **Fig 3** | Question vs Answer length scatter — shows weak correlation (r ≈ 0.1) |
| **Fig 4** | Top 5 medical categories pie chart — treatment and prevention dominate |
| **Fig 5** | Sample count per split — confirms 70/15/15 ratio |

---

## 🔧 Step-by-Step Replication

### Step 1 — VPC & Networking (AWS Console)

Create the following resources via the AWS Console. All resources are prefixed with `25cdkg-` per the course naming policy.

| Resource | Name | Configuration |
|---|---|---|
| **VPC** | `25cdkg-vpc` | CIDR: 10.0.0.0/16 |
| **Public Subnet A** | `25cdkg-public-1` | 10.0.1.0/24 (us-east-1a) — EC2 |
| **Public Subnet B** | `25cdkg-public-2` | 10.0.2.0/24 (us-east-1b) — EMR |
| **Internet Gateway** | `25cdkg-igw` | Attached to VPC |
| **Route Table** | `25cdkg-public-rt` | Route `0.0.0.0/0` → IGW; associated to both subnets |

#### Security Groups

**EMR SG** — inbound:
- Port 22 (SSH) from your IP
- All traffic from EC2 SG (for internal communication)

**EC2 SG** — inbound:
- Port 22 (SSH) from your IP
- Port 3000 (OpenWebUI) from `0.0.0.0/0`
- Port 11434 (Ollama API) from EC2 SG only

### Step 2 — S3 Bucket & Data Upload

```bash
# Create bucket
aws s3 mb s3://25cdkg-medical-qa --region us-east-1b

# Upload raw dataset
aws s3 cp train.csv s3://25cdkg-medical-qa/raw/train.csv

# Verify upload
aws s3 ls s3://25cdkg-medical-qa/raw/
```

### Step 3 — EMR Cluster & PySpark Preprocessing

#### 3a. Launch EMR Cluster

```bash
aws emr create-cluster \
  --name "25cdkg-emr" \
  --release-label emr-7.0.0 \
  --applications Name=Spark \
  --instance-groups \
    InstanceGroupType=MASTER,InstanceType=m5.xlarge,InstanceCount=1 \
    InstanceGroupType=CORE,InstanceType=m5.xlarge,InstanceCount=2 \
  --ec2-attributes \
    KeyName=25cdkg-key,SubnetId=<subnet-id-for-public-2> \
  --use-default-roles \
  --region us-east-1b
```

#### 3b. Run Preprocessing Notebook

Open `Data_Preprocessing.ipynb` and run all cells in order:

1. Load CSV from S3
2. Filter and clean records
3. Format as ChatML instruction pairs
4. Split 70/15/15
5. Save `train.jsonl`, `val.jsonl`, `test.jsonl` to S3

```bash
# Verify output
aws s3 ls s3://25cdkg-medical-qa/processed/
aws s3 ls s3://25cdkg-medical-qa/eda/
aws s3 ls s3://25cdkg-medical-qa/logs/
```

#### 3c. Terminate EMR Cluster

```bash
# Terminate cluster immediately after use
aws emr terminate-clusters --cluster-ids j-XXXXXXXXXXXXX
```

> ⚠️ **Note:** The EMR cluster must be terminated immediately after preprocessing. Screenshot of terminated cluster available in `AWS_Images/emr_terminated.png`.

### Step 4 — Model Fine-Tuning

#### Model Selection

| Field | Details |
|---|---|
| **Model** | Qwen 2.5 1.5B Instruct |
| **Parameters** | 1.56 billion |
| **Source** | Unsloth / HuggingFace |
| **License** | Apache 2.0 |
| **Quantization (training)** | NF4 4-bit (BitsAndBytes) |
| **Quantization (export)** | q4_k_m (GGUF for Ollama deployment) |

#### Why this model

- Under 10B parameters — fits on a single GPU as recommended in the project resource guide
- Apache 2.0 license — allows commercial use and modification
- Strong instruction-following capability with native ChatML template support
- 4-bit quantization reduces VRAM from 6 GB to 1.5 GB — practical for Colab and EC2
- Unsloth library provides optimized LoRA training with 2× speedup

#### Fine-Tuning Workflow

1. Open `Fine_Tuning.ipynb` on a GPU machine
2. Download processed `train.jsonl` and `val.jsonl` from S3
3. Run all cells in order: *Load Model → Base Response → Add LoRA → Train → Evaluate → Export GGUF*
4. Upload GGUF to S3 or transfer directly to EC2

#### LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

#### Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./medical_qa_model_lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    warmup_steps=50,
    logging_steps=25
)
```

### Step 5 — EC2 Deployment

#### 5a. Launch EC2 Instance

| Parameter | Value |
|---|---|
| **AMI** | Ubuntu 22.04 LTS |
| **Instance Type** | g4dn.xLarge |
| **Subnet** | `25cdkg-public-1` (10.0.1.0/24) |
| **Security Group** | EC2 SG |
| **Key Pair** | `25cdkg-key` |
| **Storage** | 50 GB gp3 |

#### 5b. Connect to EC2

```bash
ssh -i 25cdkg-key.pem ubuntu@<ec2-public-ip>
```

#### 5c. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama
```

#### 5d. Load Fine-Tuned Model

```bash
# Download GGUF from S3
aws s3 cp s3://25cdkg-medical-qa/model/medical-qa.gguf ./medical-qa.gguf

# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./medical-qa.gguf
SYSTEM "You are a professional medical assistant. Provide accurate, helpful medical information based on your training."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Register model with Ollama
ollama create medical-qa -f Modelfile

# Verify
ollama list
```

#### 5e. Run OpenWebUI

```bash
# Install Docker
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl enable docker && sudo systemctl start docker

# Run OpenWebUI — auto-restarts on reboot via --restart always
sudo docker run -d \
  --name open-webui \
  --restart always \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main

# Verify container is running
sudo docker ps
```

**Access the interface at:** `http://<25cdkg-ec2-public-ip>:3000`

1. Create an admin account on first visit
2. Select `medical-qa` model from the dropdown
3. Start asking medical questions

#### Auto-Start Configuration

- **OpenWebUI:** `--restart always` flag ensures Docker container starts automatically on EC2 reboot
- **Ollama:** Installed as a systemd service — auto-starts on boot

#### 5f. Test via cURL API

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "medical-qa",
  "prompt": "What are the symptoms of diabetes?",
  "stream": false
}'
```

### Step 6 — Teardown (After Submission)

> ⚠️ **All resources must be terminated after use to avoid charges on the shared AWS account.**

```bash
# 1. Terminate EC2 instance
aws ec2 terminate-instances --instance-ids i-XXXXXXXXX

# 2. Terminate EMR cluster (if not auto-terminated)
aws emr terminate-clusters --cluster-ids j-XXXXXXXXX

# 3. Delete S3 bucket contents and bucket
aws s3 rb s3://25cdkg-medical-qa --force

# 4. Delete VPC resources (security groups, subnets, route tables, IGW, VPC)
```

> All resources have been terminated. Screenshots available in `AWS_Images/`.

---

## ✅ Model Test Results

### Sample Outputs (Post Fine-Tuning)

**Q: What are the symptoms of pneumonia?**
> The most common symptom of bacterial pneumonia is a cough with phlegm or mucus that produces a wet sound when tapped on the chest, fever, and chills. Viral pneumonia often has no specific signs other than flu-like symptoms such as high fever, shaking chills, and extreme fatigue.

**Q: What is the treatment for asthma?**
> Medications can help prevent and control symptoms of asthma, including bronchodilators and corticosteroids. Bronchodilators work by relaxing and widening the airways so you can breathe easier. Corticosteroid inhalers reduce swelling in the lining of your airways.

**Q: What are the side effects of ibuprofen?**
> Ibuprofen can cause stomach upset, heartburn, and diarrhea. In rare cases, people take too much ibuprofen at one time and may develop bleeding problems or ulcers in their digestive tract.

**Q: How can I prevent heart disease?**
> You can take steps to lower your risk of getting heart disease, including maintaining a healthy diet, regular exercise, avoiding smoking, and managing blood pressure and cholesterol levels.

**Q: How is tuberculosis diagnosed?**
> A health care provider may diagnose active TB disease by using tests including chest X-ray, sputum culture, and tuberculin skin test (TST).

---

## 💰 Cloud Infrastructure Cost Summary

| Service                  | Service Total | April 2026* | May 2026* |
|--------------------------|---------------|-------------|-----------|
| *Total costs*          | *$2.51*     | *$2.51*   | *$0.00* |
| Elastic MapReduce        | $1.22         | $1.22       | -         |
| Elastic Load Balancing   | $0.65         | $0.65       | -         |
| EC2-Other                | $0.30         | $0.30       | $0.00     |
| VPC                      | $0.30         | $0.30       | $0.00     |
| EC2-Instances            | $0.02         | $0.02       | $0.00     |
| S3                       | $0.01         | $0.01       | $0.00     |
| Secrets Manager          | $0.00         | $0.00       | $0.00     |
| CloudShell               | $0.00         | $0.00       | -         |
| Glue                     | $0.00         | $0.00       | $0.00     |
| Key Management Service   | $0.00         | $0.00       | -         |
| SNS                      | $0.00         | $0.00       | -         |
| SQS                      | $0.00         | $0.00       | -         |
| Tax                      | $0.00         | -           | $0.00     |
| Data Transfer            | -$0.00        | -           | -$0.00    |

### By Service (Actual — AWS Cost Explorer, April 2026)

> For a minimal reproduction (EMR + EC2 + S3 only), expected cost is approximately **$15–20 USD**. All resources terminated after use. Cost data from AWS Cost Explorer screenshots in `AWS_Images/`.

---

## ⚠️ Files Not in This Repository

The following files are generated during execution and are too large for GitHub (>100 MB):

| File / Folder | Size | How to Recreate |
|---|---|---|
| `medical_qa_model_lora/` | ~100 MB | Run `Fine_Tuning.ipynb` — "Save Model" cell |
| `medical_qa_gguf/` | ~1 GB | Run GGUF export cell in `Fine_Tuning.ipynb` |
| `unsloth_compiled_cache/` | ~200 MB | Generated automatically by Unsloth during training |
| `S3/` | ~50 MB | `aws s3 cp s3://25cdkg-medical-qa/ S3/ --recursive` |

---

## 💻 Hardware Requirements

| Stage | Minimum | Used in This Project |
|---|---|---|
| **Preprocessing** | Any CPU (2 GB RAM) | EMR m5.xlarge × 3 |
| **Fine-Tuning** | 16 GB VRAM GPU | NVIDIA RTX 5000 Ada (32 GB) |
| **Deployment** | 4 GB RAM (CPU inference) | EC2 g4dn.xLarge |

---

## ✅ Expected Output at Each Stage

| Stage | Output | How to Verify |
|---|---|---|
| **Preprocessing** | `processed/*.jsonl` on S3 | `aws s3 ls s3://25cdkg-medical-qa/processed/` |
| **Fine-Tuning** | `medical_qa_model_lora/` | Contains `adapter_model.safetensors` |
| **GGUF Export** | `*.gguf` file |
| **Ollama Load** | Model registered | `ollama list` shows `medical-qa` |
| **Web Interface** | Browser accessible | Open `http://EC2_IP:3000` |

---

## 🔒 Security Note

> ⚠️ The `.pem` SSH key file is **NOT** included in this repository for security reasons. To reproduce the deployment, generate your own key pair in the AWS EC2 Console.

---

## 🏷️ Resource Naming Reference

| Resource | Name |
|---|---|
| **VPC** | `25cdkg-vpc` |
| **Internet Gateway** | `25cdkg-igw` |
| **Public Subnet A (EC2)** | `10.0.1.0/24` |
| **Public Subnet B (EMR)** | `10.0.2.0/24` |
| **Route Table** | `25cdkg-public-rt` |
| **EMR Security Group** | `EMR SG` |
| **EC2 Security Group** | `EC2 SG` |
| **S3 Bucket** | `25cdkg-medical-qa` |
