# using the python3.8 image as the base image
FROM python:3.8

# Install dependencies using pip
RUN pip install gradio requests fastapi uvicorn

# Copy the app.py file from the host machine to the /app/ directory in the container
COPY app.py /app/

# Changing the working directory to /app/
WORKDIR /app

# set the default command to run when the container starts
# this will start the uvicorn server with the app:app module, enable auto-reloading,
# and make it accessible externally on 0.0.0.0:3000
CMD uvicorn app:app --reload --host 0.0.0.0 --port 3000
