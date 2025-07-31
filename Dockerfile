# Use a slim, official Python base image for a small footprint
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# --- Install Python Dependencies ---
# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python libraries using the lean requirements file
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
# Copy the rest of your application code into the container
COPY . .

# --- Run the Application ---
# Tell Docker to expose the port Render will use
EXPOSE 10000

# The command to run when the container starts.
# We use --host 0.0.0.0 to make it accessible from outside the container.
# Render provides the PORT in an environment variable, which is best practice.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]