# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔽 Install unstructured with PDF dependencies
RUN pip install "unstructured[pdf]"

# Copy all source code
COPY . .

# Expose port (Cloud Run default is 8080)
ENV PORT=8080
EXPOSE 8080

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
