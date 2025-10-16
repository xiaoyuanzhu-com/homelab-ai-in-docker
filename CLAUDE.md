# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based homelab AI project intended to run in Docker (based on project naming). Currently in early scaffolding stage.

## Development Setup

- **Python Version**: 3.13 (specified in `.python-version`)
- **Package Manager**: Uses `pyproject.toml` for dependency management (uv or pip compatible)

## Running the Application

```bash
python main.py
```

Currently prints a simple hello message.

## Project Structure

- `main.py` - Entry point with basic main() function
- `pyproject.toml` - Python project configuration and dependencies
- `.python-version` - Specifies Python 3.13 requirement

## Notes

The project name suggests Docker integration but no Docker configuration files exist yet. When implementing Docker support, typical files would include `Dockerfile` and `docker-compose.yml`.
