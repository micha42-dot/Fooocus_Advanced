import os
import sys


def update(repo):
    import pygit2

    if repo.head_is_detached or repo.head_is_unborn or repo.state() != pygit2.GIT_REPOSITORY_STATE_NONE:
        return 'Update skipped: no active branch or a Git operation is in progress.'
    branch = repo.head
    repo.remotes['origin'].fetch()
    remote_commit = repo.revparse_single(f'refs/remotes/origin/{branch.shorthand}')

    merge_result, _ = repo.merge_analysis(remote_commit.id)

    if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
        return 'Already up-to-date.'
    if not merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
        return 'Update skipped: the local and remote branches have diverged.'
    if repo.status(untracked_files='no'):
        return 'Update skipped: commit or stash local changes first.'

    # Check out against the old HEAD so conflicts leave the branch at its original commit.
    repo.checkout_tree(remote_commit, strategy=pygit2.GIT_CHECKOUT_SAFE | pygit2.GIT_CHECKOUT_DONT_OVERWRITE_IGNORED)
    branch.set_target(remote_commit.id)
    return 'Update succeeded.'


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root)
    os.chdir(root)
    try:
        import pygit2
        repo = pygit2.Repository(root, flags=pygit2.GIT_REPOSITORY_OPEN_NO_SEARCH)
        try:
            print(update(repo))
        finally:
            repo.free()
    except Exception as exc:
        print(f'Update failed: {exc}')
    import launch
