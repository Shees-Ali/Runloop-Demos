import os
import logging
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from runloop_api_client import Runloop
from mcp.server.fastmcp import FastMCP
import openai

# Load environment variables from .env file
load_dotenv()

# Constants
DATABASE_FILE = "Chinook_Sqlite.sqlite"
RUNLOOP_API_KEY_ENV = "RUNLOOP_API_KEY"
DEVBOX_ID_ENV = "DEVBOX_ID"
OPENAI_API_KEY_ENV = "OPENAI_API"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Runloop client
runloop_client = None

def get_runloop_client():
    """Create and return a Runloop client instance."""
    global runloop_client
    if runloop_client is None:
        api_key = os.environ.get(RUNLOOP_API_KEY_ENV)
        if not api_key:
            logger.error(f"Environment variable {RUNLOOP_API_KEY_ENV} not set.")
            raise ValueError(f"Environment variable {RUNLOOP_API_KEY_ENV} not set.")
        runloop_client = Runloop(bearer_token=api_key)
    return runloop_client

def create_devbox():
    """Create a Devbox on Runloop.ai and return its ID."""
    devbox_name = "DemoDevbox"
    
    try:
        devbox = runloop_client.devboxes.create_and_await_running(
            name=devbox_name,
            setup_commands=[
                "sudo apt-get update",
                "sudo apt-get install -y python3-pip",
                "pip3 install pandas matplotlib",
                "pip3 install openai",
                "pip3 install runloop-api-client",
                "pip3 install mcp",
                "pip3 install python-dotenv",
                "pip3 install sqlite3"
            ],
            launch_parameters={
                "after_idle": {
                    "idle_time_seconds": 1800,
                    "on_idle": "suspend"
                },
            }
        )
        logger.info(f"Devbox created: {devbox.id}")
        os.environ[DEVBOX_ID_ENV] = devbox.id
        return devbox.id
    except Exception as e:
        logger.error(f"Failed to create Devbox: {e}")
        raise

def upload_files_to_devbox(devbox_id):
    """Upload the SQLite database to the Devbox."""
    try:
        with open(DATABASE_FILE, "rb") as db_file:
            runloop_client.devboxes.upload_file(
                devbox_id,
                path="/home/user/Chinook_Sqlite.sqlite",
                file=db_file
            )
        logger.info(f"Database {DATABASE_FILE} uploaded successfully.")

        with open("client.py", "rb") as client:
            runloop_client.devboxes.upload_file(
                devbox_id,
                path="/home/user/client.py",
                file=client
            )
        logger.info(f"Client file uploaded successfully.")

        with open("server.py", "rb") as server:
            runloop_client.devboxes.upload_file(
                devbox_id,
                path="/home/user/server.py",
                file=server
            )
        logger.info(f"Server file uploaded successfully.")

        
    except Exception as e:
        logger.error(f"Failed to upload database: {e}")
        raise

def main():
    """Main function to coordinate the entire workflow."""
    try:
        if not os.path.exists(DATABASE_FILE):
            logger.error(f"Database file {DATABASE_FILE} not found.")
            return

        # Initialize Runloop client
        runloop_client = get_runloop_client()

        # Check or create Devbox
        devbox_id = os.environ.get(DEVBOX_ID_ENV)
        if not devbox_id:
            logger.info("No Devbox ID found. Creating a new Devbox.")
            devbox_id = create_devbox()

        devbox = runloop_client.devboxes.retrieve(devbox_id)
        # Check if the devbox is suspended
        if devbox and devbox.status == "suspended":
            logger.info("Devbox is suspended. Resuming...")
            runloop_client.devboxes.resume(devbox_id)
            devbox = runloop_client.devboxes.await_running(devbox_id)
            logger.info(f"Devbox resumed: {devbox.id}")

        # Upload files
        logger.info("Uploading files to Devbox...")
        upload_files_to_devbox(devbox_id)        

        runloop_client.devboxes.retrieve(devbox_id)

        # Run the client in devbox
        logger.info("Running the client script in the Devbox...")
        exec_result = runloop_client.devboxes.execute_sync(devbox_id, command="python client.py Chinook_Sqlite.sqlite")
        # Print stdout
        print(exec_result)        

    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")
        raise

if __name__ == "__main__":
    main()
