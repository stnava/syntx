import re
import os
import subprocess
import sys

def get_current_version():
    with open("pyproject.toml", "r") as f:
        content = f.read()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)

def increment_version(version_str):
    parts = version_str.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version_str}")
    major, minor, patch = map(int, parts)
    patch += 1
    return f"{major}.{minor}.{patch}"

def update_file(filename, pattern, replacement):
    with open(filename, "r") as f:
        content = f.read()
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        raise ValueError(f"Pattern not found in {filename}")
    with open(filename, "w") as f:
        f.write(new_content)
    print(f"Updated {filename}")

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: {res.stderr}")
        sys.exit(res.returncode)
    print(res.stdout)
    return res.stdout

def main():
    old_version = get_current_version()
    new_version = increment_version(old_version)
    print(f"Bumping version from {old_version} to {new_version}")

    # Update pyproject.toml
    update_file(
        "pyproject.toml",
        r'(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{new_version}"'
    )

    # Update src/syntx/__init__.py
    update_file(
        "src/syntx/__init__.py",
        r'(__version__\s*=\s*)"[^"]+"',
        rf'\g<1>"{new_version}"'
    )

    # Git operations
    run_command(["git", "add", "pyproject.toml", "src/syntx/__init__.py"])
    
    commit_msg = f"release: bump version to v{new_version}"
    run_command(["git", "commit", "-m", commit_msg])
    
    tag_name = f"v{new_version}"
    run_command(["git", "tag", "-a", tag_name, "-m", f"Release version {tag_name}"])
    
    print(f"Successfully released and tagged {tag_name}")

if __name__ == "__main__":
    main()
