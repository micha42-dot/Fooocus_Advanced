import json
import os
import tempfile
import threading
from pathlib import Path

from modules import flags


SAMPLER_CONFIG_KEY = 'default_sampler'
SCHEDULER_CONFIG_KEY = 'default_scheduler'

_CONFIG_LOCK = threading.Lock()


class ConfigPersistenceError(RuntimeError):
    pass


def _read_user_config(config_path):
    path = Path(config_path)
    if path.is_symlink():
        raise ConfigPersistenceError('Refusing to replace a symbolic-link config file.')
    if not path.exists():
        return {}

    try:
        with path.open('r', encoding='utf-8') as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigPersistenceError(
            'The existing config file could not be read as JSON.'
        ) from exc

    if not isinstance(config, dict):
        raise ConfigPersistenceError('The existing config file must contain a JSON object.')
    return config


def _write_user_config(config_path, config):
    path = Path(config_path)
    if path.is_symlink():
        raise ConfigPersistenceError('Refusing to replace a symbolic-link config file.')

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f'.{path.name}.',
            suffix='.tmp',
            dir=str(path.parent),
        )
    except OSError as exc:
        raise ConfigPersistenceError('Could not prepare the config file update.') from exc

    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, 'w', encoding='utf-8', newline='\n') as config_file:
            json.dump(config, config_file, indent=4)
            config_file.write('\n')
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temp_path, path)
        temp_path = None
    except OSError as exc:
        raise ConfigPersistenceError('Could not write the config file.') from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def save_sampling_defaults(config_path, sampler, scheduler):
    if sampler not in flags.sampler_list:
        raise ConfigPersistenceError(f'Unknown sampler: {sampler}')
    if scheduler not in flags.scheduler_list:
        raise ConfigPersistenceError(f'Unknown scheduler: {scheduler}')

    with _CONFIG_LOCK:
        config = _read_user_config(config_path)
        config[SAMPLER_CONFIG_KEY] = sampler
        config[SCHEDULER_CONFIG_KEY] = scheduler
        _write_user_config(config_path, config)


def reset_sampling_defaults(config_path):
    with _CONFIG_LOCK:
        config = _read_user_config(config_path)
        sampler_removed = SAMPLER_CONFIG_KEY in config
        scheduler_removed = SCHEDULER_CONFIG_KEY in config
        config.pop(SAMPLER_CONFIG_KEY, None)
        config.pop(SCHEDULER_CONFIG_KEY, None)
        if sampler_removed or scheduler_removed:
            _write_user_config(config_path, config)
        return sampler_removed or scheduler_removed
