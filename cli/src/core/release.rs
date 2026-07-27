//! The published releases this binary comes from. `install.sh` derives the same URLs and asset names.

/// This build's version; `update base` moves the checkout to the matching tag.
pub fn current() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The tag whose scripts this binary expects: its own, unless `VIZFOLD_VERSION` pins another.
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

/// Asset name as the release workflow publishes it. Linux only, as `install.sh` also insists.
pub fn asset(arch: &str) -> String {
    format!("vizfold-linux-{arch}")
}

pub fn asset_url(tag: &str, asset: &str) -> String {
    format!(
        "https://github.com/{}/releases/download/{tag}/{asset}",
        repo()
    )
}

/// Newest tag, off the `/releases/latest` redirect: no token or rate limit, and `status` asks every run.
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

/// `.../releases/tag/v0.5.0` -> `v0.5.0`; with no releases GitHub lands on `/releases`, yielding nothing.
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
    use super::{asset, tag_from_release_url, version_line};

    /// A mismatch would download nothing on the platform install.sh bootstrapped.
    #[test]
    fn asset_names_match_the_bootstrap_installer() {
        assert_eq!(asset("x86_64"), "vizfold-linux-x86_64");
        assert_eq!(asset("aarch64"), "vizfold-linux-aarch64");
    }

    #[test]
    fn the_tag_comes_off_the_latest_release_redirect() {
        assert_eq!(
            tag_from_release_url(
                "https://github.com/AI2Science/vizfold-foundation/releases/tag/v0.5.0\n"
            ),
            Some("v0.5.0".to_owned())
        );
        assert_eq!(
            tag_from_release_url("https://github.com/AI2Science/vizfold-foundation/releases"),
            None
        );
    }

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
