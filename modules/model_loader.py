import os
from urllib.parse import urlparse
from typing import Optional


_model_downloads_enabled = True


def set_model_downloads_enabled(enabled: bool):
    global _model_downloads_enabled
    _model_downloads_enabled = bool(enabled)


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        if not _model_downloads_enabled:
            raise FileNotFoundError(
                'Automatic model downloads are disabled and a required file is missing: '
                f'{cached_file}'
            )
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file
