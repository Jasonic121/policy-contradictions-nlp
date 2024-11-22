FROM mcr.microsoft.com/devcontainers/python:0-3.10

# Install system certificates
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for SSL certificate handling
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
ENV PYTHONHTTPSVERIFY=1

# Install Python dependencies
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --upgrade pip && \
    pip install --trusted-host pypi.python.org \
                --trusted-host files.pythonhosted.org \
                --trusted-host pypi.org \
                -r /tmp/pip-tmp/requirements.txt && \
    rm -rf /tmp/pip-tmp
