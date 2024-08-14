# Base image
FROM apify/actor-python:3.11

# Copy requirements.txt
COPY requirements.txt ./

# Install dependencies and NLTK data
RUN echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Installing dependencies:" \
 && pip install -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze \
 && echo "Downloading NLTK data:" \
 && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Copy the remaining files
COPY . ./

# Specify how to launch the actor
CMD ["python3", "-m", "src"]
