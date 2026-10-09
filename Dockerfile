# Use an official stabilized Python workspace runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Establish internal workspace directory structure
WORKDIR /app

# Pull app files into the virtual image container environment
COPY . /app

# Force sequential package builds matching identical video framework requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit streamlit-webrtc mediapipe tensorflow opencv-python numpy SpeechRecognition

# Expose web platform connectivity stream port
EXPOSE 8501

# Automatically launch the processing pipeline engine
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]