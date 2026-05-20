FROM python:3.10.7-slim

# 1. Grab the compiled `uv` binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /cdss-sepsis
# 2. Copy dependencies
COPY requirements.txt .

# 3. Fast install into the system Python
RUN uv pip install --system --no-cache-dir -r requirements.txt

# 4. Copy the shared ML logic and the client script
COPY ./src /cdss-sepsis/src
COPY ./client /cdss-sepsis/client

# Run the client script
CMD ["python", "-m", "client.main"]