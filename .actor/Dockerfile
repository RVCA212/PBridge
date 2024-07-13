FROM apify/actor-python:3.11

# Install necessary dependencies including Git and Curl
RUN apt-get update && apt-get install -y git curl build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Ensure Rust is available in the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the requirements file into the image
COPY requirements.txt ./

# Upgrade pip to the latest version and install dependencies from requirements.txt
RUN echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Upgrading pip:" \
 && pip install --upgrade pip \
 && echo "Installing dependencies:" \
 && pip install -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze

# Copy the rest of the application code into the image
COPY . ./

# Specify the command to run the application
CMD ["python3", "-m", "src"]

