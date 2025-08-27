# DIP Project: English-to-Thai Machine Translation for Product Categories  
*(WIPO Nice Classification Benchmarking)*

## 📌 Project Overview
This project focuses on **English → Thai machine translation (En2Th)** with domain-specific emphasis on **product and service categories** defined by the **WIPO Nice Classification** system.  
The main goal is to evaluate and improve the translation quality for category-specific terminology, ensuring translations are accurate, consistent, and context-aware for intellectual property and business use.

## 🎯 Objectives
- Benchmark multiple **Large Language Models (LLMs)** on En2Th translation tasks.  
- Explore **fine-tuning** and **Retrieval-Augmented Generation (RAG)** methods for domain adaptation.  
- Provide insights into translation quality across **different WIPO Nice product categories**.  

## ⚙️ Methodology
1. **Data Preparation**  
   - Source: WIPO Nice classification dataset (product & service categories).  
   - Preprocessed into English–Thai parallel pairs.  

2. **Approaches**  
   - **Baseline Translation:** Zero/few-shot prompts with general-purpose LLMs.  
   - **Fine-Tuning:** Domain-specific fine-tuning on product-category text.  
   - **RAG (Retrieval-Augmented Generation):** Incorporating WIPO classification documents as external knowledge for context-aware translations.  

## 🛠️ Tech Stack
- **Fine-tuning:** unsloth
- **RAG:** faiss, langchain
- **Inference:** vllm
- **Evaluation:** CER, WER, BLEU, METEOR
