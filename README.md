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
#### Team Leader Email - sunkari.apoorvaQ@gmail.com

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
   7. python
   8. langchain (intel developer cloud)
   9. open api optimizer
   10. Fast API
   
### Step-by-Step Code Execution Instructions:
  1. clone the git repository
  2. use the transactions.csv dataset provided in the transactions folders of git repo for training and validating fraud detection model.
  3. The code base for the deployed hugging face model is under models folder.
  4. open https://huggingface.co/spaces/blaze999/llama-2-RAG UI interface chatbot for recommendations on provided/input question
  
### Future Scope:
1.This raw model is trained with minimal pdf content data for portfolio management , which helps providing invetsment recommendations on relevant data.
In future, model needs be trained with multiple investment related information for being more precise.
2. Infuse chatgpt to manage customer interface and case management
3.fine tune llama-2 model to improvise response of the RAG model.
