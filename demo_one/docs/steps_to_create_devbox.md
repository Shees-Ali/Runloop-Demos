# Steps to Create a Devbox on Runloop.ai

Follow these steps to create a Devbox on Runloop.ai and obtain its ID.

## Step 1: Set Up Your Environment

1. Ensure you have the `runloop_client` library installed.
2. Set up logging to capture information and errors.
3. Define the environment variable `DEVBOX_ID_ENV` to store the Devbox ID.

## Step 2: Define the Devbox Creation Function

Create a function `create_devbox` to handle the creation of the Devbox:

```python
def create_devbox():
    """Create a Devbox on Runloop.ai and return its ID."""
    devbox_name = "DemoDevbox"
    
    try:
        devbox = runloop_client.devboxes.create_and_await_running(name=devbox_name, launch_parameters={
            "after_idle": {
                "idle_time_seconds": 1800,
                "on_idle": "suspend"
            }
        })
        logger.info(f"Devbox created: {devbox.id}")
        os.environ[DEVBOX_ID_ENV] = devbox.id
        return devbox.id
    except Exception as e:
        logger.error(f"Failed to create Devbox: {e}")
        raise
```

## Step 3: Execute the Function

1. Call the `create_devbox` function in your main script or application.
2. Handle any exceptions that may occur during the Devbox creation process.

## Step 4: Verify the Devbox Creation

1. Check the logs to ensure the Devbox was created successfully.
2. Verify that the `DEVBOX_ID_ENV` environment variable is set with the correct Devbox ID.

By following these steps, you will be able to create a Devbox on Runloop.ai and retrieve its ID for further use in your development environment.