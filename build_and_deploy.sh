#!/bin/bash
# Build and deploy script for LunguageScore
#
# Usage:
#   ./build_and_deploy.sh            # Build only (check dist/)
#   ./build_and_deploy.sh --deploy   # Build + upload to PyPI
#   ./build_and_deploy.sh --test     # Build + upload to TestPyPI
#
# Requirements:
#   pip install build twine
#
# PyPI Authentication (API token recommended):
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-<your-api-token>
#
# Before deploying a new version, create a git tag:
#   git tag v1.0.0
#   git push origin v1.0.0

set -e

echo "=== LunguageScore Build Script ==="

# ── Version check ──────────────────────────────
CURRENT_VERSION=$(python -c "from lunguage_score._version import __version__; print(__version__)" 2>/dev/null || echo "unknown")
echo "Current version (from _version.py): $CURRENT_VERSION"

if [[ "$CURRENT_VERSION" == *"dev"* ]]; then
    echo ""
    echo "WARNING: Version contains 'dev' — this means no git tag is set."
    echo "To publish a clean release, run:"
    echo "  git tag v1.0.0"
    echo "  git push origin v1.0.0"
    echo "Then re-run this script."
    echo ""
    if [ "$1" = "--deploy" ] || [ "$1" = "--test" ]; then
        read -p "Continue with dev version? [y/N] " confirm
        [[ "$confirm" == [yY] ]] || exit 1
    fi
fi

# ── Clean previous builds ──────────────────────
echo "Cleaning previous build artifacts..."
rm -rf build/ dist/ lunguage_score.egg-info/

# ── Install build dependencies ─────────────────
pip install --upgrade build twine --quiet

# ── Build ──────────────────────────────────────
echo "Building sdist and wheel..."
python -m build

echo ""
echo "Build artifacts:"
ls -lh dist/

# ── Validate with twine ────────────────────────
echo ""
echo "Checking package with twine..."
twine check dist/*

# ── Deploy ─────────────────────────────────────
if [ "$1" = "--deploy" ]; then
    echo ""
    echo "Uploading to PyPI..."

    if [ -z "$TWINE_USERNAME" ] || [ -z "$TWINE_PASSWORD" ]; then
        echo "ERROR: Set TWINE_USERNAME and TWINE_PASSWORD before deploying."
        echo "  export TWINE_USERNAME=__token__"
        echo "  export TWINE_PASSWORD=pypi-<your-api-token>"
        exit 1
    fi

    twine upload dist/* --non-interactive
    echo "✓ Deployed to PyPI: https://pypi.org/project/lunguage-score/"

elif [ "$1" = "--test" ]; then
    echo ""
    echo "Uploading to TestPyPI..."

    if [ -z "$TWINE_USERNAME" ] || [ -z "$TWINE_PASSWORD" ]; then
        echo "ERROR: Set TWINE_USERNAME and TWINE_PASSWORD before deploying."
        echo "  export TWINE_USERNAME=__token__"
        echo "  export TWINE_PASSWORD=pypi-<your-testpypi-api-token>"
        exit 1
    fi

    twine upload --repository testpypi dist/* --non-interactive
    echo "✓ Deployed to TestPyPI: https://test.pypi.org/project/lunguage-score/"
    echo ""
    echo "Install from TestPyPI:"
    echo "  pip install --index-url https://test.pypi.org/simple/ lunguage-score"

else
    echo ""
    echo "Build complete. To deploy:"
    echo "  ./build_and_deploy.sh --test    # Upload to TestPyPI first"
    echo "  ./build_and_deploy.sh --deploy  # Upload to PyPI"
fi

echo ""
echo "Done!"
