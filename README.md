# 🚗 Car Description — End-to-End Fine-Tuning & Serving Pipeline

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Base Model](https://img.shields.io/badge/Model-Qwen2.5--3B--Instruct-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Teacher Model](https://img.shields.io/badge/Teacher-Qwen2.5--7B-orange.svg)](https://ollama.com/library/qwen2.5:7b)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A self-contained, end-to-end Google Colab pipeline ([`car_description_pipeline.ipynb`](car_description_pipeline.ipynb)) that distills structured automotive extraction knowledge from `Qwen2.5-7B` into `Qwen2.5-3B-Instruct` using **QLoRA** (LLaMA-Factory), validates the output against a Pydantic schema, publishes the adapter to Hugging Face Hub, serves it via **vLLM**, and performs headless load testing with **Locust**.

---

## 📋 Pipeline Stages

The pipeline in [`car_description_pipeline.ipynb`](car_description_pipeline.ipynb) consists of 15 sequential stages:

| # | Stage | Description |
|---|---|---|
| **0** | **GPU Check** | Verifies GPU availability (Colab T4 15 GB VRAM recommended). |
| **1** | **Configuration** | Sets hyperparameters, repository IDs, prompt templates, and paths. |
| **2** | **Environment** | Installs pinned dependencies and LLaMA-Factory. |
| **3** | **Credentials** | Loads `HF_TOKEN` and `WANDB_API_KEY` securely from Colab Secrets. |
| **4** | **Teacher Server** | Installs and spins up a local Ollama server running `qwen2.5:7b`. |
| **5** | **Schema & Prompts** | Defines Pydantic schema (`CarDetails`) and extraction prompts. |
| **6** | **Dataset** | Loads raw car listing dataset (`fashxp/cars-description`). |
| **7** | **Distillation** | Uses teacher model to generate structured JSON labels. |
| **8** | **Validate & Split** | Validates JSON against schema and creates train/val splits. |
| **9** | **Training** | Trains `Qwen2.5-3B-Instruct` with QLoRA using LLaMA-Factory. |
| **10** | **Evaluation** | Calculates per-field exact-match metrics on held-out test data. |
| **11** | **Publish** | Pushes fine-tuned LoRA adapter and dataset to Hugging Face Hub. |
| **12** | **vLLM Serving** | Serves the fine-tuned adapter using high-throughput vLLM. |
| **13** | **Load Testing** | Evaluates latency and throughput using headless Locust tests. |
| **14** | **Cleanup** | Shuts down background servers (vLLM, Ollama) and frees resources. |

---

## 🚀 How to Run in Google Colab

1. **Upload / Open in Colab**:
   Open [`car_description_pipeline.ipynb`](car_description_pipeline.ipynb) in [Google Colab](https://colab.research.google.com/).
2. **Enable GPU**:
   Go to `Runtime` > `Change runtime type` > Select **T4 GPU** (or better).
3. **Set Secrets (🔑 icon on left toolbar)**:
   - `HF_TOKEN`: Hugging Face write token (required for base model download and Hub upload).
   - `WANDB_API_KEY`: *(Optional)* Weights & Biases API key for experiment tracking.
4. **Run Cells**:
   Execute cells top-to-bottom, in order.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
