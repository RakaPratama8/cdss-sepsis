# Use the official lightweight Python 3.10 image
FROM python:3.10.7-slim

# 1. Grab the compiled `uv` binary directly from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /cdss-sepsis

# 2. Copy dependencies
COPY requirements.txt .

# 3. Use `uv pip` instead of standard `pip`
# We use the --system flag because we are installing directly into the Docker 
# container's global Python environment, bypassing the need for a virtualenv.
RUN uv pip install --system --no-cache-dir -r requirements.txt

# 4. Copy the rest of your code
# (Change the second path depending on if this is the client, server, or interface)
COPY ./src /cdss-sepsis/src
COPY ./server /cdss-sepsis/server 

# Expose ports and run the command (Example for the Server)
EXPOSE 8000 8080
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]