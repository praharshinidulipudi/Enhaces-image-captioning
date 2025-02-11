1. Clone the file to the IDE

2. Create and Activate Virtual Environment
   - Open terminal or command prompt
   - Run the following command to create a virtual environment:
     python -m venv venv
   - Activate the virtual environment:
     - On Windows:
       venv\Scripts\activate
     - On macOS/Linux:
       source venv/bin/activate

3. Install Requirements
   - Run the following command to install required dependencies:
     pip install torch>=2.2.0 torchvision>=0.17.0 transformers>=4.37.2 Pillow>=10.2.0 streamlit>=1.31.0 gTTS>=2.5.0 pandas>=2.2.0 plotly>=5.18.0 python-dotenv>=1.0.1 numpy>=1.26.3 requests>=2.31.0 protobuf>=4.25.2 nltk>=3.8.1 rouge-score>=0.1.2 pycocoevalcap

4. For First-Time Setup:
   - Run NLTK setup script:
     python nltk_setup.py
   - Run the NLTK test script:
     python test_nltk.py

5. Run the main application:
   - Execute the main script:
     python main.py

6. Access the Streamlit App:
   - After running main.py, a Streamlit link will appear in the terminal.
   - Copy and paste the link into your browser to access the application.
