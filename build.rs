use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
    println!("cargo:rerun-if-changed=.git/packed-refs");

    // ── Fork-specific: build timestamp + ODB crate version ──
    let build_time = Command::new("date")
        .arg("+%Y-%m-%d %H:%M %Z")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=FLOWSURFACE_BUILD_TIME={build_time}");

    let lock = std::fs::read_to_string("Cargo.lock").unwrap_or_default();
    let odb_version = extract_package_version(&lock, "opendeviationbar-core")
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=FLOWSURFACE_ODB_VERSION={odb_version}");

    // ── Upstream: git SHA + release tag detection ──
    if let Some(git_sha) = git_output(&["rev-parse", "--short=7", "HEAD"]) {
        println!("cargo:rustc-env=FLOWSURFACE_GIT_SHA={git_sha}");
    }

    let pkg_version = env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let is_release_tag = git_head_has_release_tag(&pkg_version);
    println!(
        "cargo:rustc-env=FLOWSURFACE_IS_RELEASE_TAG={}",
        if is_release_tag { "true" } else { "false" }
    );

    let is_official_release = is_official_release_build();
    println!(
        "cargo:rustc-env=FLOWSURFACE_IS_OFFICIAL_RELEASE={}",
        if is_official_release { "true" } else { "false" }
    );
}

fn extract_package_version(lock_content: &str, package_name: &str) -> Option<String> {
    let target = format!("name = \"{package_name}\"");
    let mut lines = lock_content.lines();
    while let Some(line) = lines.next() {
        if line.trim() == target
            && let Some(version_line) = lines.next()
        {
            let version_line = version_line.trim();
            if let Some(version) = version_line
                .strip_prefix("version = \"")
                .and_then(|s| s.strip_suffix('"'))
            {
                return Some(version.to_string());
            }
        }
    }
    None
}

fn is_official_release_build() -> bool {
    if !env_var_is_truthy("FLOWSURFACE_OFFICIAL_RELEASE") {
        return false;
    }
    if !env_var_is_truthy("GITHUB_ACTIONS") {
        return false;
    }
    let workflow = env::var("GITHUB_WORKFLOW").unwrap_or_default();
    let event = env::var("GITHUB_EVENT_NAME").unwrap_or_default();
    workflow == "Release" && event == "workflow_dispatch"
}

fn env_var_is_truthy(name: &str) -> bool {
    let Ok(value) = env::var(name) else {
        return false;
    };
    value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let value = stdout.trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

fn git_head_has_release_tag(pkg_version: &str) -> bool {
    let Some(tags) = git_output(&["tag", "--points-at", "HEAD"]) else {
        return false;
    };
    let prefixed = format!("v{pkg_version}");
    tags.lines()
        .any(|tag| {
            let value = tag.trim();
            value == pkg_version || value == prefixed
        })
}
