🏥 MedAssist — Cloud-Based Medical QA Chat Assistant

CISC 886 — Cloud Computing | Queen's University NetID: 25cdkg

Student: Salma Essam

A cloud-based conversational chatbot designed to act as a Professional Medical Assistant. It leverages a fine-tuned Qwen 2.5 1.5B Instruct LLM trained on 17,023 structured medical Q&A records derived from the Comprehensive Medical Q&A Dataset, deployed entirely on AWS infrastructure.

🚀 Quickstart (TL;DR)

Step 1 → AWS Console              # Create VPC, Subnet, SG, S3
Step 2 → aws s3 cp train.csv      # Upload raw data to S3
Step 3 → EMR Cluster (PySpark)    # Preprocess 38K records → JSONL
Step 4 → GPU Machine              # LoRA fine-tune → export GGUF
Step 5 → EC2 Instance             # Ollama + OpenWebUI (Docker)
Step 6 → Teardown                 # Terminate all resources



Full commands for each step are in the sections below.

🏗️ System Architecture

Data Flow:

Raw medical Q&A dataset uploaded from Kaggle to S3

EMR cluster (m5.xlarge × 3) runs PySpark preprocessing in Public Subnet (10.0.2.0/24)

Processed JSONL files saved back to S3 via HTTPS (port 443)

Fine-tuning notebook trains the model using LoRA (PEFT) on GPU hardware

Fine-tuned model exported to GGUF format for efficient inference

GGUF model loaded into Ollama (LLM Runner) on EC2 in Public Subnet A (10.0.1.0/24)

OpenWebUI sends LLM requests to Ollama on the same EC2 host

NAT traffic routed through Internet Gateway (25cdkg-igw) with route 0.0.0.0/0

Users access the medical assistant through their browser via port 3000

EC2 communicates with S3 via S3 API (port 443) for data and model downloads

📁 Repository Structure

.
├── README.md                       # This file
├── .gitignore                      # Excludes large model files and keys
├── Data_Preprocessing.ipynb        # PySpark preprocessing notebook (Section 4)
├── Fine_Tuning.ipynb               # Model fine-tuning notebook (Section 5)
├── Complete_Code_MQA.ipynb         # Combined complete pipeline notebook
├── install_deps.txt                # Python dependencies list
├── AWS_Images/                     # AWS Console & architecture screenshots
│   ├── architecture_diagram.png    # System architecture diagram
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
├── EDA_Images/                     # Exploratory Data Analysis figures
│   ├── fig1_question_length.png
│   ├── fig2_label_balance.png
│   ├── fig3_q_vs_a_scatter.png
│   ├── fig4_top5_categories.png
│   └── fig5_split_counts.png
└── Model_Images/                   # Training and evaluation figures
    ├── training_curves.png
    └── evaluation_metrics.png



📋 Prerequisites

|

| Requirement | Version / Notes |
| AWS Account | Region: us-east-1 (N. Virginia) |
| Python | 3.10+ |
| Apache Spark | 3.x (on EMR — no local install needed) |
| GPU (for fine-tuning) | ≥16 GB VRAM (local GPU or Google Colab T4 free tier) |
| AWS CLI | Configured with project credentials (aws configure) |
| Docker | Required on EC2 for OpenWebUI |

Python Dependencies

# Preprocessing
pip install pyspark kagglehub matplotlib seaborn pandas numpy

# Fine-tuning (requires CUDA GPU)
pip install unsloth trl peft accelerate bitsandbytes transformers datasets

# Evaluation
pip install rouge-score nltk scikit-learn



Or install all at once:

pip install -r install_deps.txt



Note: The fine-tuning notebook requires a CUDA-compatible GPU with ≥16 GB VRAM. Google Colab (T4 GPU, free tier) is sufficient as an alternative.

IAM Requirements

| Policy | Purpose |
| AmazonS3FullAccess | Upload/download data and model artefacts |
| AmazonEC2FullAccess | Launch and manage EC2 instances |
| AmazonEMRFullAccess | Create and terminate EMR clusters |
| IAMFullAccess (or scoped) | Attach instance profiles to EMR/EC2 |

EMR also requires a service role (AmazonEMR-ServiceRole) and an EC2 instance profile (AmazonEMR-InstanceProfile) — these are created automatically when you launch EMR via the console for the first time.

📊 Dataset

| Field | Details |
| Name | Comprehensive Medical Q&A Dataset |
| Source | Kaggle |
| License | CC BY 4.0 |
| Raw Samples | 38,088 |
| Post-ETL Samples | 17,023 |
| Columns | qtype, Question, Answer |

Sample (Verbatim):

Question Type: susceptibility
Q: Who is at risk for Lymphocytic Choriomeningitis (LCM)?
A: LCMV infections can occur after exposure to fresh urine, droppings,
   saliva, or nesting materials from infected rodents. Transmission may
   also occur when these materials are directly introduced into broken
   skin, the nose, the eyes, or the mouth, or presumably, via the bite
   of an infected rodent. Person-to-person transmission has not been
   reported, with the exception of vertical transmission from infected
   mother to fetus, and rarely, through organ transplantation.



Preprocessing Pipeline

| Step | Operation | Purpose |
| 1 | Lowercase + trim | Consistent text format across all records |
| 2 | Drop nulls | Remove rows with missing Question or Answer |
| 3 | Word count filtering | Q ≥ 5 words, 10 ≤ A ≤ 500 words |
| 4 | Whitespace cleanup | Collapse \s+ → single space |
| 5 | Deduplication | Remove exact duplicate (Question, Answer) pairs |
| 6 | ChatML formatting | Wrap in `< |
| 7 | Character-length filter | Q > 20 chars, A > 40 chars |

Data Leakage Prevention

Split performed after all preprocessing is complete.

Random split with fixed seed=42 for full reproducibility.

No overlap between train, validation, and test sets.

Train / Validation / Test Split

| Split | Samples | Percentage |
| Train | ~12,033 | 70% |
| Validation | ~2,507 | 15% |
| Test | ~2,483 | 15% |

EDA Figures (5 total — available in EDA_Images/)

| Figure | Description |
| Fig 1 | Question word count distribution — most questions are 5–20 words |
| Fig 2 | Top 10 question types (label balance) — identifies dominant medical categories |
| Fig 3 | Question vs Answer length scatter — shows weak correlation (r ≈ 0.1) |
| Fig 4 | Top 5 medical categories pie chart — treatment and prevention dominate |
| Fig 5 | Sample count per split — confirms 70/15/15 ratio |

🛠️ Step-by-Step Replication

Step 1 — VPC & Networking (AWS Console)

Create the following resources via the AWS Console. All resources are prefixed with 25cdkg- per the course naming policy.

| Resource | Name | Configuration |
| VPC | 25cdkg-vpc | CIDR: 10.0.0.0/16 |
| Public Subnet A | 25cdkg-public-1 | 10.0.1.0/24 (us-east-1a) — EC2 |
| Public Subnet B | 25cdkg-public-2 | 10.0.2.0/24 (us-east-1b) — EMR |
| Internet Gateway | 25cdkg-igw | Attached to VPC |
| Route Table | 25cdkg-public-rt | 0.0.0.0/0 → IGW |

Security Groups:

EMR Security Group: EMR SG

| Port | Source | Purpose |
| 22 | My IP | SSH access to EMR master node |
| Intra-cluster | Self | EMR master ↔ core node communication |

EC2 Security Group: EC2 SG

| Port | Source | Purpose |
| 22 | My IP only | SSH access for administration |
| 3000 | 0.0.0.0/0 | OpenWebUI browser access |
| 11434 | 127.0.0.1 | Ollama API (internal only — secured) |
| 443 (outbound) | 0.0.0.0/0 | HTTPS to S3 API for data/model download |

Design Justification:

10.0.0.0/16 CIDR: Provides 65,536 IPs — sufficient for all project resources with room for expansion.

Two public subnets in different AZs: EMR in Subnet B, EC2 in Subnet A — logical separation of preprocessing and deployment.

Internet Gateway + public route table: Required for S3 access (HTTPS 443), package downloads, and end-user browser access.

Port 3000 open to 0.0.0.0/0: Allows browser access to the chat interface from any location for demonstration.

Port 11434 restricted to localhost: Ollama API only accessible from OpenWebUI running on the same EC2 instance.

Port 22 restricted to My IP: SSH access limited to the developer's IP address for security.

Outbound 443: EC2 and EMR both communicate with S3 via HTTPS (S3 API port 443).

Step 2 — Upload Raw Data to S3

# Create S3 bucket
aws s3 mb s3://25cdkg-medical-qa --region us-east-1

# Download dataset from Kaggle (requires Kaggle account)
# Then upload to S3:
aws s3 cp train.csv s3://25cdkg-medical-qa/raw/train.csv



Step 3 — Data Preprocessing on AWS EMR

3a. EMR Cluster Configuration

| Setting | Value |
| Cluster Name | 25cdkg-medical-qa-preprocessing |
| EMR Release | emr-7.x |
| Master Node | 1 × m5.xlarge |
| Core Nodes | 2 × m5.xlarge |
| Region | us-east-1 |
| VPC / Subnet | 25cdkg-vpc / Public Subnet B (10.0.2.0/24) |
| Security Group | EMR SG |
| EC2 Key Pair | 25cdkg-finetuning-cloud-project |
| Auto-terminate | Enabled |

3b. Upload Preprocessing Notebook to S3

aws s3 cp Data_Preprocessing.ipynb s3://25cdkg-medical-qa/scripts/



3c. PySpark Preprocessing Pipeline

The notebook Data_Preprocessing.ipynb performs:

# Load data from S3
df = spark.read.csv("s3://25cdkg-medical-qa/raw/train.csv",
                     header=True, inferSchema=True)

# 1. Lowercase and trim all text columns
df_clean = df.withColumn("Question", lower(trim(col("Question")))) \
             .withColumn("Answer", lower(trim(col("Answer")))) \
             .withColumn("qtype", lower(trim(col("qtype"))))

# 2. Drop nulls
df_clean = df_clean.dropna(subset=["Question", "Answer"])

# 3. Add word counts and filter outliers
df_eda = df_clean.withColumn("q_len", size(split(col("Question"), " "))) \
                 .withColumn("a_len", size(split(col("Answer"), " ")))

df_filtered = df_refined.filter(
    (col("a_len") >= 10) & (col("a_len") <= 500) & (col("q_len") >= 5)
)

# 4. Clean whitespace and deduplicate
df_final_clean = df_filtered \
    .withColumn("Answer", regexp_replace(col("Answer"), r"\s+", " ")) \
    .withColumn("Question", regexp_replace(col("Question"), r"\s+", " ")) \
    .dropDuplicates(["Question", "Answer"])

# 5. Apply ChatML template for Qwen 2.5 Instruct
df_templated = df_final_clean.withColumn("text", concat(
    lit("<|im_start|>system\nYou are a professional medical assistant..."),
    lit("<|im_start|>user\n"), col("Question"), lit("<|im_end|>\n"),
    lit("<|im_start|>assistant\n"), col("Answer"), lit("<|im_end|>")
))

# 6. Split 70/15/15 and save to S3
train_df, val_df, test_df = df_final.randomSplit([0.7, 0.15, 0.15], seed=42)
train_df.select("text").coalesce(1).write.mode("overwrite") \
    .json("s3://25cdkg-medical-qa/processed/train")
val_df.select("text").coalesce(1).write.mode("overwrite") \
    .json("s3://25cdkg-medical-qa/processed/val")
test_df.select("Question", "Answer").coalesce(1).write.mode("overwrite") \
    .json("s3://25cdkg-medical-qa/processed/test_qa")



3d. Verify Output and Terminate

# Verify output files on S3
aws s3 ls s3://25cdkg-medical-qa/processed/ --recursive

# Expected output:
# processed/train/part-00000-*.json    (~12,000 samples)
# processed/val/part-00000-*.json      (~2,500 samples)
# processed/test_qa/part-00000-*.json  (~2,500 samples)

# Terminate cluster immediately after use
aws emr terminate-clusters --cluster-ids j-XXXXXXXXXXXXX



⚠️ Note: The EMR cluster must be terminated immediately after preprocessing. Screenshot of terminated cluster available in AWS_Images/emr_terminated.png.

Step 4 — Model Fine-Tuning

Model Selection

| Field | Details |
| Model | Qwen 2.5 1.5B Instruct |
| Parameters | 1.56 billion |
| Source | Unsloth / HuggingFace |
| License | Apache 2.0 |
| Quantization (training) | NF4 4-bit (BitsAndBytes) |
| Quantization (export) | q4_k_m (GGUF for Ollama deployment) |

Why this model:

Under 10B parameters — fits on a single GPU as recommended in the project resource guide.

Apache 2.0 license — allows commercial use and modification.

Strong instruction-following capability with native ChatML template support.

4-bit quantization reduces VRAM from 6 GB to 1.5 GB — practical for Colab and EC2.

Unsloth library provides optimized LoRA training with 2× speedup.

Fine-Tuning Workflow

Open Fine_Tuning.ipynb on a GPU machine.

Download processed train.jsonl and val.jsonl from S3.

Run all cells in order: Load Model → Base Response → Add LoRA → Train → Evaluate → Export GGUF.

Upload GGUF to S3 or transfer directly to EC2.

Hyperparameter Table

| Hyperparameter | Value | Reasoning |
| LoRA Rank (r) | 16 | Balances VRAM usage with expressive power for domain adaptation |
| LoRA Alpha | 16 | Standard scaling factor applied to LoRA weight updates |
| LoRA Dropout | 0 | No dropout — dataset large enough to prevent overfitting |
| Target Modules | q,k,v,o,gate,up,down_proj | All attention + MLP layers for comprehensive adaptation |
| Learning Rate | 2e-4 | Standard stable starting point for PEFT with AdamW |
| Batch Size | 1 | Optimized for memory efficiency on single GPU |
| Gradient Accumulation | 8 | Simulates effective batch size of 8 for stable gradients |
| Effective Batch Size | 8 | 1 × 8 = 8; provides stable gradient estimates |
| Epochs | 2 (3,010 steps) | Sufficient for medical domain adaptation |
| Warmup Steps | 50 | Gradual LR ramp-up to prevent early training instability |
| Optimizer | adamw_8bit | Memory-efficient optimizer provided natively by Unsloth |
| LR Scheduler | linear | Smooth linear decay throughout training |
| Max Sequence Length | 2048 | Covers longest medical Q&A pairs with comfortable margin |
| Trainable Params | 18,464,768 / 1,562,179,072 (1.18%) | Efficient — only 1.18% of weights updated |

Training Results

| Step | Training Loss | Validation Loss |
| 500 | 0.985 | 1.027 |
| 1000 | 0.983 | 0.988 |
| 1500 | 0.948 | 0.962 |
| 2000 | 0.893 | 0.954 |
| 2500 | 0.993 | 0.943 |
| 3000 | 0.896 | 0.938 |
| 3010 | 0.982 | 0.938 |

Validation loss decreased consistently from 1.027 → 0.938 (↓8.7%) with no signs of overfitting. Training curve available in Model_Images/training_curves.png.

Base vs Fine-Tuned Comparison

Prompt 1: "What are the symptoms of diabetes?"

| Model | Response |
| Base | Generic, brief response about diabetes |
| Fine-tuned | "The first sign that you may have prediabetes is an FBS level between 100 and less than or equal to 126. In this case, your blood sugar levels are higher than normal but not high enough for a diagnosis of diabetes." |

Prompt 2: "How is hypertension treated?"

| Model | Response |
| Base | Brief, generic answer |
| Fine-tuned | "A person with high blood pressure can delay or even prevent the need for medication by making lifestyle changes such as losing weight, eating less sodium (salt), and getting more physical activity. These changes may be all that a doctor recommends in mild cases of hypertension." |

Prompt 3: "What causes migraine headaches?"

| Model | Response |
| Base | Short, vague answer |
| Fine-tuned | "Scientists aren't sure what causes migraines, but they think it's probably related to changes in the brain and chemicals released by blood vessels in the head that trigger pain. Some people with migraines may also have an abnormality of their trigeminal nerve..." |

The fine-tuned model produces longer, more medically detailed, and more accurate responses.

Evaluation Metrics (100 test samples)

| Metric | Score |
| ROUGE-1 | 0.4199 |
| ROUGE-2 | 0.2319 |
| ROUGE-L | 0.3171 |
| BLEU | 0.1628 |

Evaluation chart available in Model_Images/evaluation_metrics.png.

Step 5 — EC2 Deployment (Ollama + OpenWebUI)

5a. Launch EC2 Instance via AWS Console

| Setting | Value |
| Name | 25cdkg-ec2 |
| AMI | Ubuntu 22.04 LTS |
| Instance Type | t2.micro / t3.micro |
| VPC / Subnet | 25cdkg-vpc / Public Subnet A (10.0.1.0/24) |
| Security Group | EC2 SG |
| Storage | 30 GB gp3 |
| Key Pair | 25cdkg-finetuning-cloud-project |

5b. SSH into EC2

ssh -i "25cdkg-finetuning-cloud-project.pem" ubuntu@<25cdkg-ec2-public-ip>



5c. Install Ollama

# Install the Ollama LLM runner
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

# Verify Ollama is running
sudo systemctl status ollama



5d. Load the Fine-Tuned Model

# Option A: Download GGUF from S3
aws s3 cp s3://25cdkg-medical-qa/models/unsloth.Q4_K_M.gguf ./medical-qa.gguf

# Option B: SCP from local machine
# scp -i "25cdkg-finetuning-cloud-project.pem" \
#     medical_qa_gguf/unsloth.Q4_K_M.gguf ubuntu@EC2_IP:~/medical-qa.gguf

# Create Modelfile for Ollama
cat << 'EOF' > Modelfile
FROM ./medical-qa.gguf
SYSTEM "You are a professional medical assistant. Answer the patient's questions accurately."
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
EOF

# Create and register model with Ollama
ollama create medical-qa -f Modelfile

# Verify model is loaded
ollama list



5e. Deploy OpenWebUI (Docker)

# Install Docker
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl enable docker && sudo systemctl start docker

# Run OpenWebUI — auto-restarts on reboot via --restart always
sudo docker run -d \
  --name open-webui \
  --restart always \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=[http://host.docker.internal:11434](http://host.docker.internal:11434) \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main

# Verify container is running
sudo docker ps



Access the interface at: http://<25cdkg-ec2-public-ip>:3000

Create an admin account on first visit.

Select medical-qa model from the dropdown.

Start asking medical questions.

Auto-Start Configuration:

OpenWebUI: --restart always flag ensures Docker container starts automatically on EC2 reboot.

Ollama: Installed as a systemd service — auto-starts on boot.

5f. Test via cURL API

curl http://localhost:11434/api/generate -d '{
  "model": "medical-qa",
  "prompt": "What are the symptoms of diabetes?",
  "stream": false
}'



Step 6 — Teardown (After Submission)

⚠️ All resources must be terminated after use to avoid charges on the shared AWS account.

# 1. Terminate EC2 instance
aws ec2 terminate-instances --instance-ids i-XXXXXXXXX

# 2. Terminate EMR cluster (if not auto-terminated)
aws emr terminate-clusters --cluster-ids j-XXXXXXXXX

# 3. Delete S3 bucket contents and bucket
aws s3 rb s3://25cdkg-medical-qa --force

# 4. Delete VPC resources (security groups, subnets, route tables, IGW, VPC)



All resources have been terminated. Screenshots available in AWS_Images/.

🧪 Model Test Results

Sample Outputs (Post Fine-Tuning)

Q: What are the symptoms of pneumonia?
A: The most common symptom of bacterial pneumonia is a cough with
   phlegm or mucus that produces a wet sound when tapped on the
   chest, fever, and chills. Viral pneumonia often has no specific
   signs other than flu-like symptoms such as high fever, shaking
   chills, and extreme fatigue.

Q: What is the treatment for asthma?
A: Medications can help prevent and control symptoms of asthma,
   including bronchodilators and corticosteroids. Bronchodilators
   work by relaxing and widening the airways so you can breathe
   easier. Corticosteroid inhalers reduce swelling in the lining
   of your airways.

Q: What are the side effects of ibuprofen?
A: Ibuprofen can cause stomach upset, heartburn, and diarrhea.
   In rare cases, people take too much ibuprofen at one time
   and may develop bleeding problems or ulcers in their
   digestive tract.

Q: How can I prevent heart disease?
A: You can take steps to lower your risk of getting heart disease,
   including maintaining a healthy diet, regular exercise, avoiding
   smoking, and managing blood pressure and cholesterol levels.

Q: How is tuberculosis diagnosed?
A: A health care provider may diagnose active TB disease by
   using tests including chest X-ray, sputum culture, and
   tuberculin skin test (TST).



💵 Cloud Infrastructure Cost Summary

By Service (Actual — AWS Cost Explorer, April 2026)

| AWS Service | Configuration | Cost (USD) |
| EC2-Instances | t2.micro + t3.micro (deployment & testing) | $11.41 |
| Elastic Load Balancing | Load balancer testing | $11.34 |
| VPC | NAT Gateway / Elastic IP | $9.56 |
| Tax | AWS-applied based on region | $5.22 |
| EC2-Other | EBS storage, data transfer | $2.45 |
| Elastic MapReduce (EMR) | m5.xlarge × 3 (preprocessing) | $2.45 |
| S3 | Standard storage (~5 GB) | $0.09 |
| Secrets Manager | — | $0.00 |
| CloudShell | — | $0.00 |
| Total |  | $42.51 |

By Instance Type

| Instance Type | Cost | Purpose |
| No instance type | $28.67 | VPC, S3, networking, tax |
| t2.micro | $5.38 | EC2 — Ollama + OpenWebUI deployment |
| t3.micro | $5.00 | EC2 — initial testing |
| m5.xlarge | $2.50 | EMR — Spark preprocessing (3 nodes) |
| r8g.xlarge | $0.95 | EC2 — model testing |
| t3.2xlarge | $0.02 | EC2 — brief testing |
| c5.4xlarge | $0.00 | Brief testing |
| Total | $42.51 |  |

For a minimal reproduction (EMR + EC2 + S3 only), expected cost is approximately $15–20 USD. All resources terminated after use. Cost data from AWS Cost Explorer screenshots in AWS_Images/.

🚫 Files Not in This Repository

The following files are generated during execution and are too large for GitHub (>100 MB):

| File / Folder | Size | How to Recreate |
| medical_qa_model_lora/ | ~100 MB | Run Fine_Tuning.ipynb — "Save Model" cell |
| medical_qa_gguf/ | ~1 GB | Run GGUF export cell in Fine_Tuning.ipynb |
| unsloth_compiled_cache/ | ~200 MB | Generated automatically by Unsloth during training |
| S3/ | ~50 MB | aws s3 cp s3://25cdkg-medical-qa/ S3/ --recursive |

💻 Hardware Requirements

| Stage | Minimum | Used in This Project |
| Preprocessing | Any CPU (2 GB RAM) | EMR m5.xlarge × 3 |
| Fine-Tuning | 16 GB VRAM GPU | NVIDIA RTX 5000 Ada (32 GB) |
| Deployment | 4 GB RAM (CPU inference) | EC2 t2.micro |

✅ Expected Output at Each Stage

| Stage | Output | How to Verify |
| Preprocessing | processed/*.jsonl on S3 | aws s3 ls s3://25cdkg-medical-qa/processed/ |
| Fine-Tuning | medical_qa_model_lora/ | Contains adapter_model.safetensors |
| GGUF Export | *.gguf file | File size ~1 GB |
| Ollama Load | Model registered | ollama list shows medical-qa |
| Web Interface | Browser accessible | Open http://EC2_IP:3000 |

🔒 Security Note

⚠️ The .pem SSH key file is NOT included in this repository for security reasons. To reproduce the deployment, generate your own key pair in the AWS EC2 Console.

🏷️ Resource Naming Reference

| Resource | Name |
| VPC | 25cdkg-vpc |
| Internet Gateway | 25cdkg-igw |
| Public Subnet A (EC2) | 10.0.1.0/24 |
| Public Subnet B (EMR) | 10.0.2.0/24 |
| Route Table | 25cdkg-public-rt |
| EMR Security Group | EMR SG |
| EC2 Security Group | EC2 SG |
| S3 Bucket | 25cdkg-medical-qa |
| EMR Cluster | 25cdkg-medical-qa-preprocessing |
| EC2 Instance | 25cdkg-ec2 |
| Ollama Model | medical-qa |
