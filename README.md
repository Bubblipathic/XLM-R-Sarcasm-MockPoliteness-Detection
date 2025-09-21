# XLM-RoBERTa with Multi-Task Learning for Sarcasm and Mock Politeness Detection

This repository contains the **source code** 

This project explores whether adding **mock politeness detection** as an auxiliary task improves sarcasm detection performance in Filipino (English,Tagalog, or code-mixed) faculty evaluation texts.

---

## 🔍 Project Overview
- **Model**: Fine-tuned [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)  
- **Tasks**:
  - **Sarcasm detection** (main task)  
  - **Mock politeness detection** (auxiliary task in MTL setup)  
- **Comparison**: Multi-Task Learning (MTL) vs. Single-Task Learning (STL)  
- **Deployment**: Tkinter desktop app (not included here due to size limits)

---

## 📂 This Repo Contains
- `[SOURCE CODE] XLM-R Sarcasm and Mock Politeness Detector.py` → main Python source code (with Tkinter UI)

---

## 🤗 Model & Demo

The trained models and demo app are uploaded on Hugging Face:  

👉 [View on Hugging Face](https://huggingface.co/Bubbli/XLM-R-Sarcasm-MockPoliteness-Detection)  


---

## 📌 Notes
- This repo contains only the source code.  
- Model weights (`.pt` files) and the executable (`.exe`) are too large to include here.  
- Please visit the Hugging Face link for the **trained models and demo**.  
