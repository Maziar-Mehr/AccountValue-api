# Use a lightweight official Python image as the base
# Python 3.11 is a good choice, 'slim-bookworm' provides a minimal Debian environment
FROM python:3.11-slim-bookworm

# Set the working directory inside the container
# This is where your application code will reside within the container
WORKDIR /app

# Copy the requirements file into the container
# This is done first to leverage Docker's build cache. If only requirements.txt changes,
# Docker won't re-run the pip install step, speeding up builds.
# '--chown=nonroot:nonroot' is a good practice for security if the base image provides a 'nonroot' user,
# ensuring files are owned by a less privileged user.
COPY --chown=nonroot:nonroot requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir: Reduces image size by not storing package caches.
# --upgrade pip: Ensures pip is up-to-date before installing other packages.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
# This includes your api.py, src/ folder, .pkl files, etc.
# The '.' at the end means "copy everything from the current host directory (.)
# into the current working directory inside the container (.)."
# Ensure your .pkl files and api.py are in the root of your GitHub repo for this to work.
COPY --chown=nonroot:nonroot . .

# Expose the port on which your FastAPI application will listen
# This signals to the Docker environment that port 8000 is used by the application.
# Render will use this to route external traffic.
EXPOSE 8000

# Define the command to run the application when the container starts
# CMD specifies the default command executed when a container is run without an explicit command.
# ["uvicorn", "api:app", ...] is the command in JSON array format.
# "api:app" refers to the 'app' object inside 'api.py'.
# "--host 0.0.0.0" makes the server listen on all network interfaces, allowing external access.
# "--port 8000" matches the EXPOSE instruction.
# "--workers 1" sets the number of Uvicorn worker processes. For simple deployments, 1 is fine.
# For production, you might consider adjusting workers based on CPU cores.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Optional: Improve security by running as a non-root user
# If your base image (python:3.11-slim-bookworm) provides a 'nonroot' user,
# uncommenting this line is a good security practice for production deployments.
# USER nonroot