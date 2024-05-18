# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the MLflow server port
EXPOSE 5000

# Run a custom script to start MLflow server and training
CMD ["bash", "-c", "mlflow server --backend-store-uri sqlite:////app/mlruns/mlflow.db --default-artifact-root /app/mlruns --host 0.0.0.0 --port 5000 & python src/train.py && bash"]
