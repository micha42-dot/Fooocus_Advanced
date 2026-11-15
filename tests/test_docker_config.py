from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class TestDockerConfig(unittest.TestCase):
    def test_gradio_compatibility_override_follows_main_install(self):
        dockerfile = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
        main_install = 'pip install --no-cache-dir -r /tmp/requirements_docker.txt'
        compatibility_install = (
            'pip install --force-reinstall --no-cache-dir '
            'Jinja2==3.1.3 "starlette<1.0.0"'
        )

        self.assertIn(compatibility_install, dockerfile)
        self.assertGreater(
            dockerfile.index(compatibility_install),
            dockerfile.index(main_install),
        )

    def test_compose_bypasses_proxies_for_gradio_self_check(self):
        compose = yaml.safe_load(
            (ROOT / 'docker-compose.yml').read_text(encoding='utf-8')
        )
        environment = compose['services']['app']['environment']

        self.assertIn('NO_PROXY=localhost,127.0.0.1,::1', environment)
        self.assertIn('HTTP_PROXY=', environment)
        self.assertIn('HTTPS_PROXY=', environment)

    def test_entrypoint_repairs_mounts_and_drops_root(self):
        entrypoint = (ROOT / 'entrypoint.sh').read_text(encoding='utf-8')

        self.assertIn('ln -sfn "$DATADIR/$1" "$ORIGINALDIR/$1"', entrypoint)
        self.assertIn('chown -R "$APP_USER:$APP_GROUP"', entrypoint)
        self.assertIn(
            'exec gosu "$APP_USER:$APP_GROUP" python launch.py "$@"',
            entrypoint,
        )


if __name__ == '__main__':
    unittest.main()
