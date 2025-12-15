# AI Content Forensics Tool

## Overview
This project is an NLP-based AI Content Forensics tool that analyzes text to determine:
- Whether the text is AI-generated or human-written
- The sentiment of the text (positive / negative)
- The formality level of the writing

The system also demonstrates how rewriting text can change AI-detection outcomes.

## Models Used
- Neural Sentiment Classifier (MLP – TensorFlow)
- Formality Classifier (Logistic Regression)
- AI vs Human Classifier (Logistic Regression)

A logistic regression sentiment baseline was evaluated during development but removed from the final system due to weaker confidence calibration and lexical bias.

## Features
- Interactive CLI application
- AI vs Human detection with probabilities
- Human-style rewriting
- Adversarial rewriting experiments
- GPT-based explanations (optional)

## How to Run
```bash
pip install -r requirements.txt
python final_app1.py
