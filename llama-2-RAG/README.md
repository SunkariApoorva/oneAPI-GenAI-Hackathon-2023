---
title: Llama 2 RAG
emoji: 👁
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
---

Ignor the above gradio based deployment (it's the initial version)
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

### Step-by-Step Code Execution Instructions through Intel IDC:
  1. connect to the Intel IDC VM using SSH
  2. git clone https://github.com/SunkariApoorva/oneAPI-GenAI-Hackathon-2023.git
  3. Install the requirements.txt
  4. use the transactions.csv dataset provided in the transactions folders of git repo for training and validating fraud detection model.
  5. The code base for the fraud detection model using AFD is updated in the git repo, above.
  6. Command to execute the streamlit app in Intel IDC: ``` python3 -m streamlit run multipage_app.py ```
  7. Port forwarding using ngrok: ``` ngrok http 8501 ```


