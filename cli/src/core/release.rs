//! The published releases this binary comes from: which one is newest, and what a release asset is
//! called on a given platform. `install.sh` bootstraps from the same URLs and derives the same
//! names, so the two must agree -- `asset_names_match_the_bootstrap_installer` holds them together.

/// This build's version. The checkout is pinned to the matching tag, so the two move together and
/// `self-update` has to move both.
pub fn current() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The release tag whose scripts this binary expects: its own, unless `VIZFOLD_VERSION` pins
/// another -- the same override `install.sh` takes when choosing which binary to download. Every
/// path that clones, moves or judges the checkout reads it here.
pub fn tag() -> String {
    match std::env::var("VIZFOLD_VERSION") {
        Ok(tag) if !tag.is_empty() => tag,
        _ => format!("v{}", current()),
    }
}

pub fn repo() -> String {
    std::env::var("VIZFOLD_REPO")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "AI2Science/vizfold-foundation".to_owned())
}

/// Release asset name, as the release workflow publishes them. `install.sh` builds this from
/// `uname`, which says `darwin` where Rust says `macos`.
pub fn asset(os: &str, arch: &str) -> String {
    let os = if os == "macos" { "darwin" } else { os };
    format!("vizfold-{os}-{arch}")
}

pub fn asset_url(tag: &str, asset: &str) -> String {
    format!(
        "https://github.com/{}/releases/download/{tag}/{asset}",
        repo()
    )
}

/// The newest published tag, read off the redirect GitHub serves for `/releases/latest`. No token
/// and no rate limit, unlike api.github.com -- and `status` asks on every run. `None` means the
/// question could not be answered here, which is not the same as "no newer release".
pub fn latest_tag() -> Option<String> {
    let url = format!("https://github.com/{}/releases/latest", repo());
    let output = std::process::Command::new("curl")
        .args([
            "-sIL",
            "--max-time",
            "3",
            "-o",
            "/dev/null",
            "-w",
            "%{url_effective}",
            &url,
        ])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| tag_from_release_url(&String::from_utf8_lossy(&output.stdout)))
        .flatten()
}

/// `https://github.com/<repo>/releases/tag/v0.5.0` -> `v0.5.0`. With no releases at all GitHub
/// lands on `/releases`, which carries no tag and correctly yields nothing.
fn tag_from_release_url(url: &str) -> Option<String> {
    let tag = url.trim().split_once("/releases/tag/")?.1.trim();
    (!tag.is_empty()).then(|| tag.to_owned())
}

/// Tags are `v0.5.0`, `CARGO_PKG_VERSION` is `0.5.0`.
pub fn version_of(tag: &str) -> &str {
    tag.trim().trim_start_matches('v')
}

/// This build against the newest release, as the `binary` health row reads.
pub fn version_line(latest: Option<&str>) -> String {
    let current = current();
    match latest.map(version_of) {
        None => format!("{current} (latest release unknown)"),
        Some(latest) if latest == current => format!("{current} (latest)"),
        Some(latest) => format!("{current} (latest {latest} -- run `vizfold self-update`)"),
    }
}

#[cfg(test)]
mod tests {
    use super::{asset, tag_from_release_url, version_line, version_of};

    /// install.sh maps `uname -s`/`uname -m` to the same names; a mismatch would download nothing
    /// on the platform the other one bootstrapped.
    #[test]
    fn asset_names_match_the_bootstrap_installer() {
        assert_eq!(asset("linux", "x86_64"), "vizfold-linux-x86_64");
        assert_eq!(asset("linux", "aarch64"), "vizfold-linux-aarch64");
        // Rust's OS name for what `uname -s` calls Darwin.
        assert_eq!(asset("macos", "aarch64"), "vizfold-darwin-aarch64");
    }

    #[test]
    fn the_tag_comes_off_the_latest_release_redirect() {
        assert_eq!(
            tag_from_release_url(
                "https://github.com/AI2Science/vizfold-foundation/releases/tag/v0.5.0\n"
            ),
            Some("v0.5.0".to_owned())
        );
        // No releases yet: the redirect stops at the index, which names no tag.
        assert_eq!(
            tag_from_release_url("https://github.com/AI2Science/vizfold-foundation/releases"),
            None
        );
        assert_eq!(tag_from_release_url(""), None);
    }

    #[test]
    fn version_of_accepts_a_tag_or_a_bare_version() {
        assert_eq!(version_of("v0.5.0"), "0.5.0");
        assert_eq!(version_of("0.5.0"), "0.5.0");
    }

    /// The three answers, against whatever version this build actually is.
    #[test]
    fn the_version_line_says_which_of_the_three_states_this_is() {
        let current = super::current();
        assert_eq!(version_line(Some(current)), format!("{current} (latest)"));
        assert_eq!(
            version_line(Some("v99.0.0")),
            format!("{current} (latest 99.0.0 -- run `vizfold self-update`)")
        );
        assert_eq!(
            version_line(None),
            format!("{current} (latest release unknown)")
        );
    }
}
