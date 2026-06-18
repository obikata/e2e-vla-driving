#!/usr/bin/env bash
# Launch the CARLA 0.9.15 server headless on the RTX 5070 Ti.
# Off-screen rendering (no window needed), low quality to leave VRAM for the policy.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# The CARLA install lives in ../carla (gitignored, downloaded separately).
CARLA_DIR="${CARLA_DIR:-$HERE/../carla}"
PORT="${1:-2000}"
cd "$CARLA_DIR"
echo "starting CARLA server from $CARLA_DIR on port $PORT (off-screen, Low quality) ..."
exec ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound \
  -carla-rpc-port="$PORT" -prefernvidia
