# Contributing

Branch from `main`, keep commits focused, open a pull request back into `main`.

## Run what CI runs

Rust (stable toolchain, edition 2024):

```bash
cd cli
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Shell and config:

```bash
for f in $(git ls-files '*.sh'); do bash -n "$f"; done
bash tests/launch_args.sh
bash tests/site_config.sh   # snapshot of every site's resolved config; -u to accept a change
bash tests/vocabulary.sh    # rejects env names outside the 19-key config schema
bash tests/link_mirror.sh   # the three uniclust30 layouts the AF2 mirrors ship
bash tests/esmfold_env.sh   # the torch build a driver may keep, and the specs it asks for
bash tests/install_rc.sh    # what the bootstrap appends to a shell rc, and appends only once
```

Python is checked with `flake8 . --select=E9,F63,F7,F82`.

## Shell fixes ship on a tag

The installed binary clones this repo pinned to its own release tag, so a change to
`install.sh`, `lib/`, `sites/` or `backends/*/install/*.sh` does not reach users on
merge — it needs a version bump and a `v*` tag. The release workflow publishes two
Linux musl assets (`vizfold-linux-x86_64`, `vizfold-linux-aarch64`); there is no
darwin build.
