# Docker Quickstart

A minimal Compose setup is provided to run the CLI app without installing Python locally.

## Prerequisites
- Docker and Docker Compose installed.

## Build the image
```bash
docker compose build
```

## Run the CLI
This starts the interactive menu (stdin/stdout attached):
```bash
docker compose run --rm app
```

## Open a shell inside the container
```bash
docker compose run --rm app sh
```
Once inside, you can manually launch the app with:
```bash
python -m src
```

Notes:
- The service mounts a named volume `phishing-email-detection-app` at `/app`. If you need to inspect or clean it on a default Ubuntu Docker setup, it resides under `/var/lib/docker/volumes/phishing-email-detection-app/_data` (rootful) or `~/.local/share/docker/volumes/phishing-email-detection-app/_data` (rootless).
- To override the default command for one-off tasks, append it after the service name, e.g. `docker compose run --rm app python -m src`.
