# 🥗 GenAI Health & Nutrition Vision Assistant

An AI-powered web application that analyzes food images and provides an approximate nutrition breakdown using multimodal Generative AI.

The application allows users to upload a food image, identify visible food items, estimate portions and calories, and view approximate protein, carbohydrates, and fat values.

> ⚠️ **Disclaimer:** This application provides AI-generated estimates for informational purposes only. It is not medical advice and does not diagnose diseases or prescribe treatment.

---

## 📌 Project Overview

The **GenAI Health & Nutrition Vision Assistant** is a multimodal AI application built using **Streamlit** and **Google Gemini Vision**.

The system accepts a food image and optionally a natural-language question. The AI analyzes the image and generates a structured nutrition response containing:

- 🍴 Identified food items
- 📏 Estimated portions
- 🔥 Estimated calorie ranges
- 🥩 Protein
- 🍚 Carbohydrates
- 🥑 Fat
- 📊 Confidence levels
- ⚠️ Uncertainty and assumptions
- 📝 Nutrition summary

The system is designed as an **assistive nutrition tool**, not as a clinical or diagnostic system.

---

## 🎯 Objectives

The main objectives of the project are:

1. Provide an easy-to-use food image upload interface.
2. Validate uploaded images before AI processing.
3. Recognize visible food items using multimodal AI.
4. Generate approximate calorie and nutrition estimates.
5. Communicate uncertainty instead of presenting false precision.
6. Allow users to provide corrections.
7. Maintain session-based analysis history.
8. Provide JSON export of analysis history.
9. Apply safety controls for medical and inappropriate requests.
10. Protect API credentials and user data.

---

## ✨ Key Features

### 📷 Food Image Upload

- Supports JPG, JPEG and PNG images.
- Displays the uploaded image before analysis.
- Checks image size and format.
- Detects invalid or corrupted images.
- Checks minimum image resolution.

### 🔍 Image Quality Validation

The application performs basic quality checks for:

- Very dark images
- Overexposed images
- Blurry images
- Images with insufficient detail

Users are warned when the image quality may affect the accuracy of the analysis.

### 🤖 AI Food Recognition

Google Gemini multimodal AI is used to analyze the uploaded food image.

The system attempts to identify only foods that are reasonably visible in the image.

It avoids intentionally inventing:

- Food items
- Ingredients
- Portions
- Calories

when the information cannot reasonably be determined.

### 🔥 Calorie Estimation

The application provides a calorie **range** instead of claiming an exact calorie value.

Example:

```text
Estimated Calories:
450 - 550 kcal
```text
Estimated Calories:
450 - 550 kcal
