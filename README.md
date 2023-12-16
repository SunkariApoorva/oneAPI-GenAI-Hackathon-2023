# oneAPI-GenAI-Hackathon-2023 - Hack2Skill

Welcome to the official repository for the oneAPI-GenAI-Hackathon-2023 organized by Hack2Skill!

## Getting Started

To get started with the oneAPI-GenAI-Hackathon-2023 repository, follow these steps:

### Submission Instruction:
  1. Fork this repository
  2. Create a folder with your Team Name
  3. Upload all the code and necessary files in the created folder
  4. Upload a **README.md** file in your folder with the below mentioned informations.
  5. Generate a Pull Request with your Team Name. (Example: submission-XYZ_team)

### README.md must consist of the following information:

#### Team Name - ATHENA
#### Problem Statement - Gen AI-Powered Customer Support Optimization
#### Team Leader Email - sunkari.apoorva@gmail.com

### A Brief of the Prototype:
  Generative AI is transforming the financial services industry by facilitating tailored customer experiences, strengthening fraud detection capabilities, and offering optimized personalized investment decisions.Generative AI, with its capacity to analyze extensive data and derive valuable insights, enables financial institutions to provide innovative services, enhance operational efficiency, and effectively mitigate risks.
  
### Tech Stack: 
   List Down all technologies used to Build the prototype
   1. LLAMA-2 (Intel developer cloud)
   2. AWS S3
   3. Amazon Fraud detector
   4. FAISS
   5. Hugging face
   6. gradio
   7. python IDC Optimized
   8. langchain (intel developer cloud)
   9. open api optimizer
   10. Fast API
   11. Streamlit
   12. Intel_extension_for_transformers from Intel oneAPI tool kit
   13. Deployment using IDC


### Step-by-Step Code Execution Instructions through Intel IDC:
  1. connect to the Intel IDC VM using SSH
  2. git clone https://github.com/SunkariApoorva/oneAPI-GenAI-Hackathon-2023.git
  3. Install the requirements.txt
  4. use the transactions.csv dataset provided in the transactions folders of git repo for training and validating fraud detection model.
  5. The code base for the fraud detection model using AFD is updated in the git repo, above.
  6. Command to execute the streamlit app in Intel IDC: ``` python3 -m streamlit run multipage_app.py ```
  7. Port forwarding using ngrok: ``` ngrok http 8501 ```


### Step-by-Step Code Execution Instructions (Ignore this, refer to execution instructions through IDC Above):
  1. clone the git repository
  2. use the transactions.csv dataset provided in the transactions folders of git repo for training and validating fraud detection model.
  3. The code base for the fraud detection model using AFD is updated in the git repo, above.
  4. GenAI Model and Code Repo: https://huggingface.co/spaces/blaze999/llama-2-RAG/tree/main
  5. Huggingface spaces Deployment URL: https://huggingface.co/spaces/blaze999/llama-2-RAG UI interface chatbot for recommendations on provided/input question
  
### Future Scope:
1.This raw model is trained with minimal pdf content data for portfolio management , which helps providing invetsment recommendations on relevant data.
In future, model needs be trained with multiple investment related information for being more precise.
2. Infuse chatgpt to manage customer interface and case management
3.fine tune llama-2 model to improvise response of the RAG model.
