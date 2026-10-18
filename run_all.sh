#!/usr/bin/env bash
# Starts the backend at once on gunicorn (a production web server). Inside it, a background job fetches the
# latest WFP prices (LIVE=1), trains every crop (fast models first, then the graph neural network), trains the
# insurance model, and re-checks WFP every REFRESH_HOURS. One worker process (it owns the background job),
# 8 threads for visitors; long timeout because a first season plan for a new place can take ~30 s.
cd "$(dirname "$0")"
exec gunicorn --chdir api --workers 1 --threads 8 --timeout 300 --bind "0.0.0.0:${PORT:-8000}" \
     --access-logfile - app:app
