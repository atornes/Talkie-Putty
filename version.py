"""Single source of the app version. The CI build overwrites this from the
git tag (see .github/workflows/build-installer.yml), so the running app's
version matches the release it was built from."""
__version__ = "0.1.0"
