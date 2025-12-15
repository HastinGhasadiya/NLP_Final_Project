# AI Content Forensics Tool

## Overview
This project is an NLP-based AI Content Forensics tool that analyzes text to determine:
- Whether the text is AI-generated or human-written
- The sentiment of the text (positive or negative)
- The formality level of the writing

The system also demonstrates how rewriting text can influence AI-detection outcomes.

## Models Used
- Neural Sentiment Classifier (MLP – TensorFlow)
- Formality Classifier (Logistic Regression)
- AI vs Human Classifier (Logistic Regression)

A logistic regression sentiment baseline was evaluated during development but removed from the final system due to weaker confidence calibration and lexical bias.

## Features
- Interactive command-line interface (CLI)
- AI vs Human classification with probabilities
- Sentiment and formality analysis
- Human-style rewriting
- Adversarial rewriting experiments
- GPT-based explanations (optional)

## Project Structure
NLP_Final_Project/
├── final_app.py
├── artifacts/
│ ├── sentiment_mlp.h5
│ ├── sentiment_mlp_vectorizer.joblib
│ ├── formality_logreg.joblib
│ └── ai_vs_human_logreg.joblib
├── README.md


## OpenAI API (Optional)
GPT-based features require an OpenAI API key.

Set the API key as an environment variable:

**Windows**
setx OPENAI_API_KEY "your_api_key_here"

If no API key is set, the application will still run using only the machine learning classifiers.

## Installation
Install the required dependencies:


python final_app.py

Type text when prompted.  
Enter `q` to quit the program.

## Notes
TensorFlow warnings during startup are expected and do not affect functionality.

## Author
Hastin Ghasadiya



