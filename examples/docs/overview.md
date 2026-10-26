# Lumen Fleet Platform Overview

Lumen Fleet is a fleet-observability and mission-replay platform for autonomous
mobile robots. Operations teams use it to watch robot health, replay missions,
search telemetry, and alert on incidents across one or many sites.

## Current release

Lumen Fleet 3.2 (build 3.2.7) was released on 14 September 2026. It is available
as a managed cloud deployment or as a self-hosted gateway for sites that cannot
send telemetry to the public internet.

## Supported robot firmware

The platform accepts telemetry from these robot families:

- Helio R2 with firmware 4.0 or newer.
- Helio R3 with firmware 4.2 or newer.
- Helio R4 with firmware 5.1 or newer.

Older firmware can connect through the compatibility bridge, but mission replay
is limited to pose and battery data. Firmware 5.1 on the R4 adds lidar summary
streams and on-device anomaly flags.

## Main components

The **Lumen Agent** runs beside the robots, buffers telemetry during network
outages, and forwards batches to the Fleet API. The **Fleet API** stores events
and serves queries. **Mission Replay** reconstructs a robot's path and sensor
state. **Insights** turns recurring incidents into saved investigations.

Lumen Fleet 3.2 introduced cursor-based event pagination, per-site rate limits,
and a new `lumen-agent doctor` command for diagnosing connectivity problems.
