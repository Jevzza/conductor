# Contributing

Thanks for wanting to help. Conductor is a hobby project, maintained when there's time, so replies
and merges may take a while.

## Reporting problems

Open an issue with the **Bug report** form and attach the logs it asks for (they're listed in
[INSTALL.md](INSTALL.md#troubleshooting)). Please don't attach music files or anything copyrighted.

## Pull requests

- Keep each pull request to one change, and say what it fixes or adds and how you tested it.
- For anything big (a new blend type, a new game bridge, changes to the trainer), open an issue first
  so we can agree on the approach.
- Code that runs on players' PCs gets extra scrutiny: `packaging/src_extra/boot.py`,
  `conductor/addlist.py`, the `.cmd` / `.sh` launchers and the mod Lua.
- Don't commit music, rendered audio, recordings, caches, trained models or personal paths. The
  `.gitignore` keeps the usual ones out.
- By contributing you agree your contribution is released under the [MIT licence](LICENSE).

## Running from source

See the README ("Running from source") and [packaging/README-build.md](packaging/README-build.md).
