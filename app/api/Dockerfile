# Use Python 3.8 as base image
FROM python:3.8

# Copy requirements file to container
COPY requirements.txt .

# Install dependencies from requirements file
RUN pip install -r requirements.txt

# Copy all files to container
COPY . .

# Expose port 5000
EXPOSE 5000

# Set command to run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
