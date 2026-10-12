"""
NEXUS Web Application Entrypoint.

Starts the FastAPI server with live frontend static files.
Access the web dashboard at: http://127.0.0.1:8000
API Swagger documentation at: http://127.0.0.1:8000/docs
"""
import os
import sys
import logging
import uvicorn
from dotenv import load_dotenv

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
os.makedirs('logs', exist_ok=True)
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler('logs/nexus.log', maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger("nexus.app")

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "8000"))


def main():
    logger.info("=" * 60)
    logger.info("STARTING NEXUS AUTONOMOUS LEARNING PLATFORM")
    logger.info("=" * 60)
    logger.info(f"Dashboard URL: http://{HOST}:{PORT}")
    logger.info(f"API Swagger:   http://{HOST}:{PORT}/docs")
    logger.info("=" * 60)

    uvicorn.run(
        "src.api.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
