# Deployment Guide for AgenticProcessGNN

This guide provides several methods for deploying and sharing the AgenticProcessGNN project.

## Method 1: Direct GitHub Access

The simplest way to share the code is via the GitHub repository:

```
https://github.com/ERPdotAI/AgenticProcessGNN
```

### For contributors:

1. Clone the repository:
   ```bash
   git clone https://github.com/ERPdotAI/AgenticProcessGNN.git
   cd AgenticProcessGNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python -m src.main
   ```

## Method 2: Docker Deployment (Recommended)

For consistent environments across different systems, use Docker:

1. Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

2. Clone the repository:
   ```bash
   git clone https://github.com/ERPdotAI/AgenticProcessGNN.git
   cd AgenticProcessGNN
   ```

3. Build and start the container:
   ```bash
   docker-compose up --build
   ```

4. Access the application at `http://localhost:8000`

## Method 3: Manual Setup

If you prefer to set up without Docker:

1. Ensure you have Python 3.9+ installed

2. Download and extract the ZIP archive 

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python -m src.main
   ```

6. Access the application at `http://localhost:8000`

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: If you encounter errors about missing packages, try:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Port Conflicts**: If port 8000 is already in use, modify the port in `src/main.py`

3. **Data Directory Issues**: Ensure the data directory exists and has write permissions:
   ```bash
   mkdir -p data/vector_store data/raw data/processed data/reference_processes
   ```

### Getting Help

For additional assistance, please create an issue on the GitHub repository or contact the maintainers. 