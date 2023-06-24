# Use the official Python image as the base image
FROM python:3.10.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container and install the dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install  -r requirements.txt --user

# Copy the application code into the container
COPY . .

# Expose port 5000 for the application
EXPOSE 5000

# Start the application
CMD ["/root/.local/bin/uvicorn", "main:app", "--host", "127.0.0.1", "--port", "5000"]