import os

win32_root = os.path.dirname(os.path.dirname(__file__))
python_embeded_path = os.path.join(win32_root, 'python_embeded')

is_win32_standalone_build = os.path.exists(python_embeded_path) and os.path.isdir(python_embeded_path)

win32_cmd = '''
.\python_embeded\python.exe -s Fooocus\entry_with_update.py {cmds} %*
pause
'''


def build_launcher():
    if not is_win32_standalone_build:
        return

    launchers = {
        'run.bat': '',
        'run_anime.bat': '--preset anime',
        'run_realistic.bat': '--preset realistic',
        'run_no_download.bat': '--disable-model-download',
    }

    for launcher_name, commands in launchers.items():
        win32_cmd_preset = win32_cmd.replace('{cmds}', commands)
        bat_path = os.path.join(win32_root, launcher_name)
        if not os.path.exists(bat_path):
            with open(bat_path, "w", encoding="utf-8") as f:
                f.write(win32_cmd_preset)
    return
