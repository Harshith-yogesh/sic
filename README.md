# PDF Summarizer and Chatbot using LLaMa2 in Streamlit
![Screenshot (481)](https://github.com/user-attachments/assets/f5c8a6d0-e9db-4113-a6d2-a44e478b0845)

## Project Overview
The PDF Summarizer Chatbot is a user-friendly application that allows you to upload PDF documents and receive concise summaries generated using advanced Large Language Models (LLMs). This project leverages the power of Natural Language Processing (NLP) to extract meaningful insights from textual data, making document analysis faster and more efficient.

## Project Purpose
This project can be a starting point for beginners who want to learn about LLMs. I use **Replicate**, which provides **free** cloud API services with open-source models like **LLaMa2**. The open-source Python framework **Streamlit** is used to deploy the model into an interactive web app. Overall, this project was made as simple as possible to help your understanding of the implementation of the LLMs project.

## Features
- **PDF Parsing**: Extract text from PDF files using PyPDF2.
- **AI-Powered Summarization**: Summaries are generated using the Llama 2 model, renowned for its state-of-the-art performance in NLP tasks.
- **Interactive User Interface**: Built with Streamlit, providing an intuitive platform for users to upload files and receive outputs.
- **Themes**: Support light and dark themes for user convenience.
- **API Integration**: Utilizes the Replicate API for seamless communication with the LLM backend.

## Getting Started
1. Clone the repository
   ```
   git clone https://github.com/0xichikawa/PDF-summarizer-chatbot-using-LLaMa2
   cd PDF-summarizer-chatbot-using-LLaMa2 
2.  Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Set Up Replicate API Key
   Obtain your Replicate API key from [replicate.com](replicate.com) and add it to `secrets.toml` file in the `.streamlit` folder:
   ```
   REPLICATE_API_TOKEN = "INSERT_YOUR_REPLICATE_API_TOKEN_HERE"
4. Run the application
   ```
   streamlit run app.py  

## Future Enhancements
- Multi-language support for summarization.
- Enhanced text extraction with OCR for scanned PDFs.
- Options for customized summary lengths and formats.

## Acknowledgments
- Meta AI for Llama 2.
- Replicate for their API services.
- Streamlit and PyPDF2 for simplifying the development process.
- Data Professor (https://github.com/dataprofessor) for tutorials and project inspiration
