import hashlib
import os
import subprocess
import tempfile

import requests
from deforum.utils.logging_config import logger
from tqdm import tqdm


def download_file_to(url: str = "",
                     destination_dir: str = "",
                     filename: str = ""):
    """


    :param url: URL to fetch
    :param destination_dir: Destination
    :param filename: Filename
    :return:
    """
    os.makedirs(destination_dir, exist_ok=True)
    filepath = os.path.join(destination_dir, filename)

    # Check if file already exists
    if os.path.exists(filepath):
        logger.info(f"File {filename} already exists in models/checkpoints/")
        return filename

    # Download file in chunks with progress bar
    logger.info(f"Downloading {filename}...")
    response = requests.get(url, stream=True, headers={'Content-Disposition': 'attachment'})
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        logger.error("ERROR: Something went wrong while downloading the file.")
    else:
        logger.info(f"{filename} downloaded successfully!")
    return os.path.join(destination_dir, filename)


def clone_if_not_exists(repo_url, local_path):
    """Clone the repository if the directory doesn't exist."""
    if not os.path.isdir(local_path):
        subprocess.run(["git", "clone", repo_url, local_path])


def checksum(filename, hash_factory=hashlib.blake2b, chunk_num_blocks=128):
    h = hash_factory()
    with open(filename, 'rb') as f:
        while chunk := f.read(chunk_num_blocks * h.block_size):
            h.update(chunk)
    return h.hexdigest()


def download_file_with_checksum(url, expected_checksum, dest_folder, dest_filename):
    os.makedirs(dest_folder, exist_ok=True)
    expected_full_path = os.path.join(dest_folder, dest_filename)
    if os.path.isdir(expected_full_path):
        raise IsADirectoryError(f"Model path is a directory: {expected_full_path}")
    if os.path.isfile(expected_full_path):
        if checksum(expected_full_path) == expected_checksum:
            return expected_full_path
        logger.warning(f"Checksum mismatch for existing model {expected_full_path}; downloading it again")

    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{dest_filename}.", suffix=".part", dir=dest_folder
    )
    os.close(file_descriptor)

    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with open(temporary_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        actual_checksum = checksum(temporary_path)
        if actual_checksum != expected_checksum:
            raise RuntimeError(
                f"Checksum mismatch for {dest_filename}: expected {expected_checksum}, "
                f"got {actual_checksum}"
            )

        os.replace(temporary_path, expected_full_path)
        return expected_full_path
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
