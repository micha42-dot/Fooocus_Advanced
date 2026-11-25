from pathlib import Path
import tempfile
import unittest

import pygit2

from entry_with_update import update


class TestUpdater(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.origin = Path(temporary.name) / 'origin'
        self.checkout = Path(temporary.name) / 'local'
        self.remote = pygit2.init_repository(self.origin, initial_head='main')
        self.addCleanup(self.remote.free)
        self.first = self.commit({'.gitignore': 'ignored.txt\n', 'app.py': 'old', 'other.txt': 'original'})
        self.local = pygit2.clone_repository(str(self.origin), str(self.checkout))
        self.addCleanup(self.local.free)

    def commit(self, files):
        index = self.remote.index
        for name, content in files.items():
            (self.origin / name).write_text(content, encoding='utf-8')
            index.add(name)
        index.write()
        author = pygit2.Signature('Tester', 'tester@example.invalid')
        parents = [] if self.remote.head_is_unborn else [self.remote.head.target]
        return self.remote.create_commit('HEAD', author, author, 'fixture', index.write_tree(), parents)

    def test_clean_fast_forward_keeps_untracked_user_files(self):
        target = self.commit({'app.py': 'new'})
        (self.checkout / 'custom.txt').write_text('mine')
        self.assertIn('succeeded', update(self.local))
        self.assertEqual(self.local.head.target, target)
        self.assertEqual((self.checkout / 'app.py').read_text(), 'new')
        self.assertEqual((self.checkout / 'custom.txt').read_text(), 'mine')
        self.assertIn('up-to-date', update(self.local))

    def test_local_worktree_and_staged_changes_survive_an_update(self):
        self.commit({'app.py': 'new'})
        path = self.checkout / 'other.txt'
        path.write_text('local edit')
        for stage in (False, True):
            with self.subTest(stage=stage):
                if stage:
                    index = self.local.index
                    index.add('other.txt')
                    index.write()
                self.assertIn('skipped', update(self.local))
                self.assertEqual(self.local.head.target, self.first)
                self.assertEqual(path.read_text(), 'local edit')
                self.assertEqual((self.checkout / 'app.py').read_text(), 'old')
                if stage:
                    self.assertEqual(self.local[self.local.index['other.txt'].id].data, b'local edit')

    def test_conflicting_user_file_does_not_advance_branch(self):
        for name in ('untracked.txt', 'ignored.txt'):
            with self.subTest(name=name):
                path = self.checkout / name
                path.write_text('mine')
                self.commit({name: 'incoming'})
                with self.assertRaises(pygit2.GitError):
                    update(self.local)
                self.assertEqual(path.read_text(), 'mine')
                self.assertEqual(self.local.head.target, self.first)
                path.unlink()
