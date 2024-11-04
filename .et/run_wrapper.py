# run_wrapper.py
import subprocess
import logging
import os

# Configure logging for output and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(command, output_path=None, env=None, timeout=3600):
    """
    Execute a shell command with enhanced error handling and logging.

    Args:
        command (str): The command to execute.
        env (dict, optional): Environment variables to use for the command.
        :param timeout:

    Returns:
        subprocess.CompletedProcess: Contains information about the executed process, including stdout and stderr.

    Raises:
        subprocess.exec_client: If command timeouts.
    """
    logger.info("Executing command: %s", command)
    try:
        if output_path:
            result = subprocess.run(
                command,
                check=False,
                env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                shell=True,
                encoding='utf-8',
                timeout=timeout,
                text=True
            )
        else:
            result = subprocess.run(
                command,
                check=True,
                env=env,
                capture_output=True,
                shell=True,
                text=True
            )

    except subprocess.TimeoutExpired as e:
        result = e
        result.stdout = str(result.stdout, 'utf-8') if result.stdout is not None else ""
        result.stderr = str(result.stderr, 'utf-8') if result.stderr is not None else ""
    except Exception as e:
        print(f"message {e.message}")

    if output_path:
        output_path = clean_path(output_path)
        os.makedirs(output_path, exist_ok=True)
        import time
        now = time.strftime("%Y%m%d-%H%M%S")
        stdout = os.path.join(output_path, 'stdout-' + now + '.txt')
        stderr = os.path.join(output_path, 'stderr-' + now + '.txt')
        with open(stdout, 'w') as f:
            f.write(f"Executing command: {command}\n")
            f.write(result.stdout)
        with open(stderr, 'w') as f:
            f.write(result.stderr)

    return result


def clean_path(output_path):
    bad_chars = [';', ':', '!', "*", '[', ']']
    for i in bad_chars:
        output_path = output_path.replace(i, '')
    return output_path
