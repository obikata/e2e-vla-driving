#!/usr/bin/env bash
# Launch CARLA 0.10 (Unreal Engine 5.5) headless. Epic quality for the demo visuals.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CARLA_DIR="${CARLA_DIR:-$HERE/../carla-ue5/Carla-0.10.0-Linux-Shipping}"
PORT="${1:-2000}"
QUALITY="${2:-Epic}"
cd "$CARLA_DIR"
echo "starting CARLA UE5 from $CARLA_DIR port $PORT quality $QUALITY ..."
exec ./CarlaUnreal.sh -RenderOffScreen -quality-level="$QUALITY" -nosound \
  -carla-rpc-port="$PORT" -prefernvidia
