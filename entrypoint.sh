#!/bin/sh
# Starts as root only to make a bind-mounted /data writable, then drops to the unprivileged
# user. If the container is started with a non-root `user:` this is a plain exec.
set -e
if [ "$(id -u)" = "0" ]; then
    mkdir -p "${DATA_DIR:-/data}" 2>/dev/null || true
    chown -R app:app "${DATA_DIR:-/data}" 2>/dev/null || true   # read-only mounts: app.py reports it
    exec gosu app "$@"
fi
exec "$@"
