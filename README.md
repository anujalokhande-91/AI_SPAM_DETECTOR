AI Spam Message Detector using GRU (Deep Learning)
🚀 Project Overview

This project is a Spam Message Detection System built using a GRU (Gated Recurrent Unit) deep learning model. The application takes a text message as input and predicts the probability (in percentage) of whether the message is Spam or Not Spam.

The system is deployed using a Flask web application, where users can interact through a simple HTML interface.

🎯 Features
📥 User-friendly web interface (HTML - index.html)
⚡ Real-time spam detection
📊 Displays probability (%) of spam vs non-spam
🧠 Deep Learning model using GRU
🔍 High accuracy (~98%)

🛠️ Tech Stack
Frontend: HTML (index.html)
Backend: Flask (Python)
Model: GRU (Deep Learning - TensorFlow/Keras)
Data Processing: Tokenization, Encoding, Padding
Model Files: .h5 (trained model), .pkl (tokenizer/encoder)

📊 Dataset Details
Total Records: 5572 messages
Columns:
label → Spam / Not Spam
message → Text message
Dataset source: Loaded via URL
Labels are predefined


⚙️ Data Preprocessing Steps
Text Cleaning
Tokenization
Integer Encoding
Sequence Padding
Label Encoding

🧠 Model Building (GRU)
Used Embedding Layer
Applied GRU (Gated Recurrent Unit)
Output layer with activation function for classification
Training Details:
Epochs: 7
Accuracy: ~98%

▶️ How to Run the Project
Step 1: Install Dependencies
pip install -r requirements.txt
Step 2: Run Flask App
python app.py
Step 3: Open Browser
http://127.0.0.1:5000/

💡 How It Works
User enters a message in the web interface
Message is preprocessed (tokenized & encoded)
Processed input is passed to GRU model
Model predicts probability
Output displayed as:
Spam %
Not Spam %

📈 Example Output
Input: "Congratulations! You won a free lottery"
Output: Spam: 95% 

Author 
Anuja Lokhande

This project is for educational purposes.
