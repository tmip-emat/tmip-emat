
======================
Installing from Source
======================

Instructions for installing TMIP-EMAT directly from source code will
be published soon.


Using Github to Fork TMIP-EMAT
==============================

When developing an implementation of TMIP-EMAT, you may find it useful
to "fork" the TMIP-EMAT repository using Github.  This will create a
clean copy of the entire repository, which you can edit and add to
as necessary for your own project.  Refer to Github for further
instructions on
`how to fork a repository <https://help.github.com/en/articles/fork-a-repo>`_.


Configuring Your Fork to Also Use TMIP-EMAT as a Remote
-------------------------------------------------------

If you want to keep your fork in sync with future changes
to the master TMIP-EMAT repository, you can do so by making
the master TMIP-EMAT repository an "upstream remote".

To do so, open 'Git Bash' (Windows) or a Terminal (Linux/MacOS),
and navigate to the location of the forked repository on your
local machine.

You can view the currently configured remote repositories for
your fork like this::

    $ git remote -v
    > origin  https://github.com/YOUR_USERNAME/tmip-emat.git (fetch)
    > origin  https://github.com/YOUR_USERNAME/tmip-emat.git (push)

To specify that you want to sync the TMIP repository with your fork::

    $ git remote add tmip https://github.com/tmip-emat/tmip-emat.git

Then, each time you want to download new content from the upstream
master repository::

    $ git fetch tmip
    > remote: Enumerating objects: 11, done.
    > remote: Counting objects: 100% (11/11), done.
    > remote: Compressing objects: 100% (2/2), done.
    > remote: Total 11 (delta 9), reused 11 (delta 9), pack-reused 0
    > Unpacking objects: 100% (11/11), done.
    > From https://github.com/tmip-emat/tmip-emat
    >    7b8c800..a9eeb0c  dev        -> tmip/dev
    >    91255d5..a9eeb0c  master     -> tmip/master

To sync with your local work, first make sure you have checked out
the branch you want to merge into (if you have more than one branch)::

    $ git checkout master
    > Switched to branch 'master'

Then you can merge the upstream changes into your local fork, which
will make your local fork's branch in sync with the TMIP repository,
without losing your local changes::

    $ git merge tmip/master
    > Updating a422352..5fdff0f
    > Fast-forward
    >  README                    |    9 -------
    >  README.md                 |    7 ++++++
    >  2 files changed, 7 insertions(+), 9 deletions(-)
    >  delete mode 100644 README
    >  create mode 100644 README.md

