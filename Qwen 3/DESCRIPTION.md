# 🚗 NuScenes-Qwen3: Multimodal Driving Scene Reasoning with Vision-Language Models

## 📌 Overview

This project fine-tunes a **multimodal Vision-Language Model (VLM)** to perform **autonomous driving scene reasoning** using synchronized camera images and LiDAR information from driving datasets.

The system trains a Qwen3-VL model to answer natural-language questions about real driving environments by jointly reasoning across multiple sensor modalities. The goal is to move beyond simple perception toward **semantic and spatial understanding**, similar to reasoning systems used in autonomous vehicles.

The notebook demonstrates an end-to-end pipeline including:

* Multimodal dataset streaming
* Structured chat-style data formatting
* Parameter-efficient fine-tuning (LoRA)
* Memory-efficient quantized training
* Supervised fine-tuning using TRL's SFTTrainer

---

## 🧠 Project Goals

* Train a multimodal reasoning assistant for autonomous driving scenarios
* Fuse multi-camera and LiDAR information into a unified representation
* Enable question-answering about driving scenes (objects, distances, environment context)
* Perform efficient fine-tuning on limited hardware using quantization + LoRA

---

## 🏗️ System Architecture

```
Dataset (NuScenes QA)
        │
        ▼
Multimodal Formatting
(Chat + Images + LiDAR)
        │
        ▼
Qwen3-VL Processor
(Tokenization + Image Encoding)
        │
        ▼
LoRA Adapter Training
(PEFT Fine-Tuning)
        │
        ▼
Supervised Fine-Tuning (TRL)
        │
        ▼
Multimodal Driving Reasoning Model
```

---

## 🔄 Pipeline Explanation

### 1️⃣ Dataset Loading

The project streams samples from a NuScenes-based QA dataset to reduce memory usage and to account for errors in dataset.

Key characteristics:

* Multi-view camera images (front, rear, side views)
* LiDAR scene information
* Natural-language driving questions
* Ground-truth reasoning answers

---

### 2️⃣ Multimodal Chat Formatting

Each sample is converted into a **chat-style conversation**:

* **System message**

  * Defines the model as a driving-scene reasoning assistant.
* **User message**

  * Contains:

    * Multiple camera images
    * LiDAR data
    * Question text
* **Assistant message**

  * Target reasoning answer.

This format aligns with instruction-tuned VLM architectures.

---

### 3️⃣ Input Processing

The `Qwen3VLProcessor`:

* Applies chat templates
* Tokenizes text prompts
* Encodes multi-image inputs
* Creates model-ready tensors

Images from multiple camera viewpoints are jointly provided to the model to encourage cross-view reasoning.

---

### 4️⃣ Memory-Efficient Model Loading

To enable training on limited GPUs:

* 4-bit quantization (BitsAndBytes)
* Gradient checkpointing
* Small batch size training

This significantly reduces VRAM requirements while maintaining performance.

---

### 5️⃣ Parameter-Efficient Fine-Tuning (LoRA)

Instead of updating the full model:

* Low-Rank Adapters (LoRA) are inserted into attention layers
* Only a small subset of parameters is trained

Benefits:

* Faster training
* Lower memory usage
* Preserves pretrained knowledge

Target modules:

```
q_proj
v_proj
```

---

### 6️⃣ Training (TRL SFTTrainer)

Supervised Fine-Tuning is performed using:

* HuggingFace TRL `SFTTrainer`
* Custom multimodal collator
* Chat-based loss masking
* Evaluation during training

The model learns to generate reasoning answers conditioned on multimodal inputs.

---

## ⚙️ Training Configuration

| Parameter           | Value                |
| ------------------- | -------------------- |
| Model               | Qwen3-VL-8B-Instruct |
| Quantization        | 4-bit NF4            |
| Training Method     | LoRA (PEFT)          |
| Batch Size          | 1                    |
| Epochs              | 1                    |
| Optimizer           | paged_adamw_32bit    |
| Max Sequence Length | 128                  |

Designed for constrained compute environments.

---

## 📂 Repository Structure (Suggested)

```
.
├── NuScenes-Qwen3.ipynb
├── README.md
├── output/              # training checkpoints
└── utils/               # optional helper scripts
```

---

## 📊 Dataset

This project uses a NuScenes-based multimodal QA dataset.

👉 **Dataset Link:**

```
https://huggingface.co/datasets/KevinNotSmile/nuscenes-qa-mini
```

Expected dataset contents:

* Multi-camera images
* LiDAR representations
* Scene reasoning questions
* Ground-truth answers

---

## ▶️ Usage

### 1. Install Dependencies

```bash
pip install transformers datasets peft trl bitsandbytes
```

---

### 2. Run Training

Open and execute:

```
NuScenes-Qwen3.ipynb
```

The notebook will:

1. Load dataset samples
2. Format multimodal conversations
3. Load quantized Qwen3-VL
4. Apply LoRA adapters
5. Train and evaluate the model

---

### 3. Evaluation

The trainer automatically performs evaluation before and during training:

```python
trainer.evaluate()
```

---

## 🧩 Key Features

✅ Multimodal reasoning (vision + LiDAR + language)
✅ Instruction-tuned chat formatting
✅ Quantized training for low VRAM GPUs
✅ Parameter-efficient fine-tuning (LoRA)
✅ Streaming dataset support
✅ Autonomous driving reasoning focus

---

## 🚧 Current Limitations

* Small dataset subset used for demonstration
* Limited training epochs
* LiDAR treated indirectly through formatted inputs
* Not optimized for real-time inference yet

---

## 🔮 Future Improvements

* Full dataset training
* True BEV (Bird’s Eye View) fusion integration
* Temporal reasoning across frames
* Multi-step chain-of-thought supervision
* ROS / autonomous stack integration
* Evaluation on real driving benchmarks

---

## 🤝 Suggestions

Suggestions and experiments are welcome! Possible areas:

* Better multimodal fusion strategies
* Improved prompt engineering
* Evaluation metrics for driving reasoning
* Efficient inference optimization
