"""Build source and Windows release ZIPs from one committed Git revision."""

import argparse
import ast
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {
    'run.bat': '',
    'run_anime.bat': ' --preset anime',
    'run_realistic.bat': ' --preset realistic',
    'run_no_download.bat': ' --disable-model-download',
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def launcher(options):
    text = rf'''@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "FOOOCUS_PYTHON=python_embeded\python.exe"
if not exist "%FOOOCUS_PYTHON%" set "FOOOCUS_PYTHON=.venv\Scripts\python.exe"
if not exist "%FOOOCUS_PYTHON%" (
    echo Python runtime missing. See INSTALL.txt for setup.
    popd
    pause
    exit /b 1
)
"%FOOOCUS_PYTHON%" -s "Fooocus\launch.py"{options} %*
set "FOOOCUS_EXIT_CODE=%ERRORLEVEL%"
popd
pause
exit /b %FOOOCUS_EXIT_CODE%
'''
    return text.replace('\n', '\r\n').encode('ascii')


def installation_notes(version):
    text = rf'''Fooocus Advanced {version}

1. Extract the ZIP and open the folder containing run.bat.
2. Place your compatible python_embeded folder next to run.bat and Fooocus.
   Alternatively, install 64-bit Python 3.12 and run these commands here:

       py -3.12 -m venv .venv
       .venv\Scripts\python.exe -m pip install packaging==24.1

3. Start run.bat. On first launch, Fooocus installs missing dependencies
   and may download the default model.

run_anime.bat and run_realistic.bat select their respective presets.
run_no_download.bat disables automatic model downloads. Place a checkpoint
in Fooocus\models\checkpoints first. Dependency installation may need internet.

UPGRADE
Close Fooocus. Copy the new Fooocus folder into your existing installation
and replace files. Update both modules and ldm_patched. Your local config.txt,
.env, checkpoints and generated images are excluded from this update.
Restart Fooocus and reload the browser.
'''
    return text.replace('\n', '\r\n').encode('ascii')


def validate_payload(source):
    excluded = {'.git', '.agents', '.codex', '.venv', '__pycache__', 'tmp', 'outputs', 'cache', 'release'}
    excluded_names = {'config.txt', 'config_modification_tutorial.txt', 'auth.json', 'ui-config.json'}
    for entry in source.infolist():
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or '..' in path.parts or excluded.intersection(path.parts):
            raise ValueError(f'Unexpected release path: {path}')
        if path.name.startswith('.env') or path.name in excluded_names:
            raise ValueError(f'Local configuration in source revision: {path}')
        if path.suffix == '.py':
            compile(source.read(entry), str(path), 'exec')
    if source.testzip() is not None:
        raise ValueError('Invalid Git source archive.')


def write_assets(output, assets):
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as rollback:
        for name, content in assets.items():
            target = output / name
            if name == 'RELEASE-NOTES.md' and target.is_file() and target.read_bytes() == content:
                continue  # Release notes can be authored directly in the output folder.
            with target.open('xb') as stream:
                rollback.callback(target.unlink)
                stream.write(content)
        rollback.pop_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default='HEAD', help='Committed Git revision to package.')
    parser.add_argument('--notes', type=Path, required=True, help='Release notes Markdown file.')
    args = parser.parse_args()
    notes = args.notes.read_bytes()
    revision = git('rev-parse', '--verify', f'{args.ref}^{{commit}}').decode().strip()
    source_bytes = git('archive', '--format=zip', revision)

    with ZipFile(io.BytesIO(source_bytes)) as source:
        validate_payload(source)
        tree = ast.parse(source.read('fooocus_version.py'))
        version = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == 'version')
        stem = 'Fooocus-Advanced-' + version.replace(' ', '-')
        folder = f'Fooocus Advanced {version}/'
        info = {'product': 'Fooocus Advanced', 'version': version, 'commit': revision}
        build_info = (json.dumps(info, indent=2) + '\n').encode('utf-8')
        extras = {name: launcher(options) for name, options in LAUNCHERS.items()}
        extras.update({'INSTALL.txt': installation_notes(version),
                       'RELEASE-NOTES.md': notes, 'BUILD-INFO.json': build_info})
        windows_buffer = io.BytesIO()
        with ZipFile(windows_buffer, 'w', compression=ZIP_DEFLATED, compresslevel=7) as windows:
            for entry in source.infolist():
                windows.writestr(folder + 'Fooocus/' + entry.filename, source.read(entry))
            for name, content in extras.items():
                windows.writestr(folder + name, content)
        windows_bytes = windows_buffer.getvalue()
        with ZipFile(io.BytesIO(windows_bytes)) as windows:
            if windows.testzip() is not None:
                raise ValueError('Invalid Windows archive.')
            for name in source.namelist():
                if windows.read(folder + 'Fooocus/' + name) != source.read(name):
                    raise ValueError(f'Windows/source mismatch: {name}')

    output = ROOT / 'release' / stem
    assets = {f'{stem}-Source.zip': source_bytes, f'{stem}-Windows.zip': windows_bytes}
    checksums = [f'{hashlib.sha256(content).hexdigest()}  {name}' for name, content in sorted(assets.items())]
    assets.update({'BUILD-INFO.json': build_info, 'RELEASE-NOTES.md': notes,
                   'SHA256SUMS.txt': ('\n'.join(checksums) + '\n').encode('ascii')})
    write_assets(output, assets)
    print(f'Packaged Git revision {revision}')


if __name__ == '__main__':
    main()
