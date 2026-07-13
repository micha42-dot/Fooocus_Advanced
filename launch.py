import os
import ssl
import sys

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(root, "cache", "torchinductor"))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

import platform
import fooocus_version

from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content
from modules import model_loader

load_file_from_url = model_loader.load_file_from_url
set_model_downloads_enabled = getattr(model_loader, 'set_model_downloads_enabled', lambda enabled: None)

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu128")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.10.0 torchvision==0.25.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.35')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

from modules import config
from modules.hash_cache import init_cache

model_downloads_disabled = (
    getattr(args, 'disable_model_download', False)
    or getattr(config, 'disable_model_download', False)
)
set_model_downloads_enabled(not model_downloads_disabled)

os.environ["U2NET_HOME"] = config.path_inpaint

os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):
    from modules.model_paths import find_file_in_folder_list, get_file_name_from_folder_list
    from modules.util import get_file_from_folder_list

    local_checkpoint_path = os.path.join(root, 'models', 'checkpoints')
    configured_paths = {
        os.path.normcase(os.path.abspath(os.path.realpath(path)))
        for path in config.paths_checkpoints
    }
    normalized_local_path = os.path.normcase(os.path.abspath(os.path.realpath(local_checkpoint_path)))
    if os.path.isdir(local_checkpoint_path) and normalized_local_path not in configured_paths:
        config.paths_checkpoints.append(local_checkpoint_path)
        print(f'[Models] Added local checkpoint fallback: {local_checkpoint_path}')

    def resolve_checkpoint(model_name):
        filename = find_file_in_folder_list(model_name, config.paths_checkpoints, recursive=True)
        if filename is None:
            return None
        return get_file_name_from_folder_list(filename, config.paths_checkpoints)

    def select_installed_checkpoint(allow_any=False):
        for model_name in [default_model] + list(previous_default_models):
            resolved_name = resolve_checkpoint(model_name)
            if resolved_name is not None:
                return resolved_name

        if allow_any:
            installed_models = config.get_model_filenames(config.paths_checkpoints)
            if installed_models:
                return installed_models[0]
        return None

    if model_downloads_disabled:
        selected_model = select_installed_checkpoint(allow_any=True)
        print('[Models] Automatic model downloads are disabled.')
        if selected_model is None:
            searched_paths = ', '.join(config.paths_checkpoints)
            raise FileNotFoundError(
                'Automatic model downloads are disabled and no checkpoint was found. '
                f'Searched: {searched_paths}'
            )
        if selected_model != default_model:
            print(f'[Models] Using installed checkpoint [{selected_model}] instead of [{default_model}].')
        return selected_model, {}

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        selected_model = select_installed_checkpoint(allow_any=True)
        print('[Models] Skipped preset model downloads.')
        if selected_model is not None:
            return selected_model, {}
        return default_model, {}

    resolved_default_model = resolve_checkpoint(default_model)
    if resolved_default_model is not None:
        if resolved_default_model != default_model:
            print(f'[Models] Found configured checkpoint as [{resolved_default_model}].')
        default_model = resolved_default_model

    if not args.always_download_new_model:
        if resolve_checkpoint(default_model) is None:
            for alternative_model_name in previous_default_models:
                resolved_alternative = resolve_checkpoint(alternative_model_name)
                if resolved_alternative is not None:
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{resolved_alternative}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = resolved_alternative
                    break

    for file_name, url in checkpoint_downloads.items():
        existing_file = find_file_in_folder_list(file_name, config.paths_checkpoints, recursive=True)
        if existing_file is not None:
            print(f'[Models] Using existing checkpoint: {existing_file}')
            continue
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_checkpoints))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        if find_file_in_folder_list(file_name, config.paths_loras, recursive=True) is not None:
            continue
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_loras))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads


config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads, config.vae_downloads)

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)

from webui import *
