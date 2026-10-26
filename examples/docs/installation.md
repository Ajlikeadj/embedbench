# Installing the Lumen Agent

The Lumen Agent is distributed as a signed container image and as a Debian
package. The container is the supported path for most deployments.

## Host requirements

- Ubuntu 22.04 or 24.04, or Red Hat Enterprise Linux 9.
- Four vCPUs and 8 GB of RAM per 25 robots.
- At least 40 GB of free disk for the local telemetry spool.
- Outbound HTTPS access to the Fleet API.

## Environment variables

The agent reads configuration from `/etc/lumen/agent.env`:

| Variable | Purpose | Default |
| --- | --- | --- |
| `LUMEN_API_URL` | Fleet API base URL | `https://api.lumen.example.com/v1` |
| `LUMEN_API_TOKEN` | Site-scoped API token | required |
| `LUMEN_SITE_ID` | Site identifier | required |
| `LUMEN_AGENT_PORT` | Local health and metrics port | `8787` |
| `LUMEN_LOG_LEVEL` | `debug`, `info`, `warn`, or `error` | `info` |

Set `LUMEN_API_URL` to the internal gateway address when the site uses an
on-premises deployment.

## Install and start

```bash
docker pull registry.example.com/lumen/agent:3.2.7
docker run -d --name lumen-agent \
  --env-file /etc/lumen/agent.env \
  -p 8787:8787 \
  registry.example.com/lumen/agent:3.2.7
```

For the on-premises gateway, allow inbound TCP 443 for the admin UI, TCP 8443
for telemetry ingestion, and TCP 8787 on the agent host for health checks.
Aggregate gateway logs are shipped over outbound TCP 6514.
