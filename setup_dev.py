#!/usr/bin/env python3
"""
Setup script for URDB County Rates development environment.
Run this after cloning the repository to set up your environment.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.11+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required, but found {version.major}.{version.minor}")
        print("   Please upgrade Python and try again.")
        return False

    print(f"✅ Python {version.major}.{version.minor} detected")
    return True


def main():
    """Set up the development environment."""
    print("🚀 Setting up URDB County Rates development environment...")
    print()

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Are you in the project root directory?")
        sys.exit(1)

    # Create virtual environment (optional but recommended)
    create_venv = input("📦 Create virtual environment? [y/N]: ").lower().strip()
    if create_venv in ['y', 'yes']:
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)

        # Activation instructions
        if sys.platform == "win32":
            activation_cmd = "venv\\Scripts\\activate"
        else:
            activation_cmd = "source venv/bin/activate"

        print(f"📝 To activate the virtual environment, run:")
        print(f"   {activation_cmd}")
        print()

    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("⚠️  Failed to install package. You may need to install dependencies manually.")
        print("   Try: pip install -r requirements.txt")

    # Create necessary directories
    directories = ["data", "output", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

    # Create sample environment file
    env_file = Path(".env.example")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# OpenEI API Key - Get yours at https://openei.org/services/api/signup/\n")
            f.write("OPENEI_API_KEY=your_api_key_here\n")
        print(f"📋 Created example environment file: {env_file}")

    # Check if CLI is working
    print()
    print("🧪 Testing CLI installation...")
    if run_command("urdb-rates info", "Testing CLI command"):
        print("✅ CLI is working correctly!")
    else:
        print("⚠️  CLI test failed. You may need to reinstall or check your PATH.")

    print()
    print("🎉 Setup complete!")
    print()
    print("📋 Next steps:")
    print("1. Get an OpenEI API key: https://openei.org/services/api/signup/")
    print("2. Set your API key: export OPENEI_API_KEY='your_key_here'")
    print("3. Try the CLI: urdb-rates info")
    print("4. Run example: python examples/sample_usage.py")
    print("5. Run tests: python tests/test_basic.py")
    print()
    print("📚 Documentation: https://github.com/your-username/urdb-utils")


if __name__ == "__main__":
    main()