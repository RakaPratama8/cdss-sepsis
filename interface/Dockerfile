FROM python:3.10.7-slim

# 1. Grab the compiled `uv` binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /cdss-sepsis

# 2. Copy dependencies
COPY requirements.txt .

# 3. Fast install into the system Python
RUN uv pip install --system --no-cache-dir -r requirements.txt

# 4. Copy ONLY the UI code
COPY ./interface /cdss-sepsis/interface

# Expose Streamlit's port
EXPOSE 8501

# Run the Streamlit dashboard
CMD ["streamlit", "run", "interface/cdss-sepsis.py", "--server.port=8501", "--server.address=0.0.0.0"]