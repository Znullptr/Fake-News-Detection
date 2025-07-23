# Fake News Detection Application

This repository contains the source code and resources for a project aimed at detecting fake news using machine learning techniques and deploying the solution as a web application.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Contributors](#contributors)

---

## Overview

Fake news can manipulate the public for political, commercial, or other purposes by spreading deliberately distorted or false information. This project classifies news articles as fake or real using machine learning algorithms. The final solution is accessible via a web interface built using Flask, enabling users to validate news articles with a simple click.

### Objectives:
1. Classify news articles into "real" or "fake."
2. Implement advanced preprocessing techniques to improve model performance.
3. Deploy the model for public usage with an easy-to-use interface.

---

## Features

- **Web Scraping:** Crawl news sites using BeautifulSoup in combination with Newspaper3k.
- **News Classification:** Classifies articles based on features extracted from their text.
- **Preprocessing Pipelines:** Includes filtering, tokenization, stop-word removal, and lemmatization.
- **Machine Learning Models:** Trained with algorithms like:
  - Passive Aggressive Classifier
  - Random Forest
  - Decision Tree
  - Naïve Bayes
- **Model Improvement:** Incorporates similarity checks, offensive word detection, and misspelled word analysis.
- **Flask Deployment:** Provides a lightweight web interface for real-time news validation.

---

## Technologies

### Programming Language:
- **Python 3.9.1**

### Libraries:
1. **Machine Learning:**
   - `scikit-learn`
   - `pandas`
   - `numpy`
2. **Natural Language Processing:**
   - `nltk`
   - `textblob`
   - `better-profanity`
3. **Web Framework:**
   - `Flask`
4. **Data Extraction:**
   - `BeautifulSoup`
   - `newspaper3k`

---

## Contributors

- **Tahar Jaafer**  
  - [LinkedIn Profile](https://linkedin.com/in/tahar-jaafer)  

