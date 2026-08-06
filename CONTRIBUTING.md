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
# Every suite, the way CI runs them -- a new tests/*.sh is picked up without editing anything.
for f in tests/*.sh; do echo "== $f"; bash "$f"; done
```

Python is checked with `flake8 . --select=E9,F63,F7,F82`.

## Shell fixes ship on a tag

The installed binary clones this repo pinned to its own release tag, so a change to
`install.sh`, `lib/`, `sites/` or `backends/*/install/*.sh` does not reach users on
merge — it needs a version bump and a `v*` tag. The release workflow publishes two
Linux musl assets (`vizfold-linux-x86_64`, `vizfold-linux-aarch64`); there is no
darwin build.
