# Use an official Python 3.9 base image with a minimal operating system version
FROM python:3.9-slim
# Define the working directory inside the container
WORKDIR /app
# Copy the requirements.txt file to the working directory
COPY app/requirements.txt . 
# Install the Python dependencies listed in requirements.txt without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt
# Copy the contents of the local app folder into the container's working directory
COPY app/ .
# Open port 5000 to allow connections to the application
EXPOSE 5000
# Define the default command to execute when the container starts
CMD ["python", "api.py"]
COPY train.py .
RUN python train.py
