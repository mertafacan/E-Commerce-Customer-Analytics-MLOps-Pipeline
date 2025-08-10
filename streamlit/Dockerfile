FROM python:3.12-slim

# Create working directory
WORKDIR /app

# Install uv package manager
RUN pip install uv

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies and install with uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy application code (app and helper module)
COPY streamlit/app.py .
COPY streamlit/model_utils.py .

# Copy Hydra configuration
COPY include/conf /app/conf

# Port required for Streamlit to run
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

