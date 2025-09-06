# 1. Start with an official Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code and the trained model
COPY . .

# 5. Expose the port the app runs on
EXPOSE 8000

# 6. Define the command to run the Uvicorn server
#    This command starts the API when the container launches.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]