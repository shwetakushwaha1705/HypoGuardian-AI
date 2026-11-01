# HypoGuardian AI: Comprehensive Deployment Guide

This guide provides step-by-step instructions for deploying HypoGuardian AI across multiple environments. Choose the option that best fits your demonstration needs:

| Deployment Method | Cost | Difficulty | Best For |
| :--- | :--- | :--- | :--- |
| **Option 1: Streamlit Community Cloud** | **100% Free** | ⭐ Very Easy | **College Viva / Project Review** (Public `.streamlit.app` link for examiners) |
| **Option 2: Hugging Face Spaces** | **100% Free** | ⭐ Very Easy | Alternative free cloud hosting with permanent URL |
| **Option 3: Docker Container** | Free locally | ⭐⭐ Intermediate | Production containerization (Render, AWS, GCP, Azure) |
| **Option 4: Local Network / Offline Viva**| Free | ⭐ Easy | Presenting inside a college lab over local Wi-Fi |

---

## Option 1: Streamlit Community Cloud (Recommended for B.Tech Viva)

Streamlit Community Cloud is the standard, free platform for hosting Streamlit applications directly from your GitHub repository. It gives you a public URL (e.g. `https://hypoguardian-ai.streamlit.app`) that examiners and professors can access on their own phones or laptops.

### Step 1: Initialize Git and Commit Your Project
Open PowerShell in your project folder (`HypoGuradian-AI`) and run:
```powershell
# 1. Initialize git repository
git init

# 2. Add all files
git add .

# 3. Create your initial commit
git commit -m "HypoGuardian AI: Complete deep learning early warning prototype"
```

### Step 2: Push to GitHub
1. Open your browser and log into [GitHub](https://github.com).
2. Click **New Repository** (green button).
3. Name the repository: `HypoGuardian-AI`.
4. Keep it **Public** (required for free Streamlit Community Cloud).
5. Do **not** initialize with README, .gitignore, or license (we already created them).
6. Click **Create repository**.
7. Copy the commands shown on GitHub and run them in your PowerShell terminal:
```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/HypoGuardian-AI.git
git push -u origin main
```

### Step 3: Connect and Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and click **Sign in with GitHub**.
2. Click **Create app** (or **New app**).
3. Select your repository: `<YOUR_GITHUB_USERNAME>/HypoGuardian-AI`.
4. Select the branch: `main`.
5. In **Main file path**, ensure it says: `app.py`.
6. (Optional) In **App URL**, customize your subdomain (e.g., `hypoguardian-ai`).
7. Click **Deploy!**

> ⏱️ *Deployment takes approximately 2–3 minutes while Streamlit installs packages and initializes your trained model. Once finished, your live app will be accessible worldwide!*

---

## Option 2: Hugging Face Spaces (Alternative Free Cloud Hosting)

Hugging Face Spaces provides free cloud hosting with 16GB RAM and 2 vCPUs on the free tier.

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Go to **Spaces** > **Create new Space**.
3. Space name: `hypoguardian-ai`.
4. Select Space SDK: **Streamlit**.
5. Select Space Hardware: **CPU Basic (Free)**.
6. Click **Create Space**.
7. Clone the repository and push your project files, or upload files directly through the web UI:
   - Upload `app.py`, `config.py`, `requirements.txt`
   - Upload `src/`, `utils/`, `models/`, `data/`, `.streamlit/`
8. The Space will build and deploy automatically within 3 minutes.

---

## Option 3: Docker Container Deployment

HypoGuardian AI includes a preconfigured [`Dockerfile`](file:///C:/Users/PC/OneDrive/Documents/Desktop/Documents/Desktop/HypoGuradian-AI/Dockerfile), [`.dockerignore`](file:///C:/Users/PC/OneDrive/Documents/Desktop/Documents/Desktop/HypoGuradian-AI/.dockerignore), and [`docker-compose.yml`](file:///C:/Users/PC/OneDrive/Documents/Desktop/Documents/Desktop/HypoGuradian-AI/docker-compose.yml).

### Method A: Using Docker CLI
```powershell
# Build the Docker image
docker build -t hypoguardian-ai .

# Run the container mapping port 8501
docker run -d -p 8501:8501 --name hypoguardian_container hypoguardian-ai
```
Open `http://localhost:8501` in your browser.

### Method B: Using Docker Compose
```powershell
docker compose up --build -d
```
To stop the container:
```powershell
docker compose down
```

### Deploying the Docker Container to Cloud Platforms
You can push this Docker image to:
- **Render.com**: Create a free Web Service with Docker runtime.
- **Railway.app**: Connect your GitHub repo and select Dockerfile.
- **Google Cloud Run / AWS ECS**: Push the image to ECR/GCR and deploy as a serverless container.

---

## Option 4: Local Network / Viva Presentation (Over College Wi-Fi)

If your college presentation lab does not have reliable internet access, you can host the application on your laptop and allow evaluators to access it from their laptops or smartphones over the local college Wi-Fi:

### Step 1: Bind to All Network Interfaces
Open PowerShell and run:
```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Step 2: Find Your Laptop's Local IP Address
In another PowerShell window, run:
```powershell
ipconfig
```
Look for **IPv4 Address** under your active Wi-Fi adapter (e.g., `192.168.1.45`).

### Step 3: Connect from Other Devices
Ensure the other device (evaluator's phone or laptop) is connected to the same Wi-Fi network. Open a web browser on that device and navigate to:
```
http://192.168.1.45:8501
```
*(Replace `192.168.1.45` with your actual IPv4 address).*

---

## Option 5: 1-Click Local Launch (Windows)

For quick offline demonstrations without remembering terminal commands:
- Simply double-click [`run_app.bat`](file:///C:/Users/PC/OneDrive/Documents/Desktop/Documents/Desktop/HypoGuradian-AI/run_app.bat) in the project folder.
- The script checks Python and launches the web app on `http://localhost:8501` automatically.
