# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-05-07

### Added

- Treatment of constant features during selection.
- `__version__` exposed via `mufs._version`.
- `CHANGELOG.md`.

### Changed

- Migrated packaging from `setup.py` to `pyproject.toml` (PEP 621).
- Modernized CI: matrix testing on Python 3.11, 3.12, 3.13 and 3.14 across
  Ubuntu and macOS runners.
- Updated SonarQube scan and quality-gate actions to pinned major versions.
- Updated the feature selection computation and refreshed the test suite.
- Bumped `scikit-learn` minimum to `>=1.8.0`.

### Removed

- Support for Python 3.10 (now requires Python `>=3.11`).
- Legacy `setup.py` / `requirements*.txt` based build flow.

## [0.1.3] - 2022-05-19

### Changed

- README updates.
- Internal cleanup of `setup.py` and `__init__.py`.

## [0.1.2] - 2021-10-28

### Added

- IWSS (Incremental Wrapper-based Subset Selection) implementation.
- SonarQube scanner integration in CI.

### Fixed

- Correlation-based Feature Selection (CFS) merit formula.

## [0.1.1] - 2021-08-02

### Added

- Initial public release: Fast Correlation-Based Filter (FCBF) and
  Correlation-based Feature Selection (CFS).
