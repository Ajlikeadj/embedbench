# Fleet API essentials

The Fleet API base URL is `https://api.lumen.example.com/v1`. All requests use
an `Authorization: Bearer <token>` header. Tokens are scoped to one site and a
set of permissions.

## Events and limits

A single telemetry event may be at most **5 MB** after compression. Batch
ingestion accepts up to **500 events** per request. The events endpoint uses
cursor pagination: pass `limit` (maximum 200) and the opaque `cursor` returned
by the previous page. Do not construct cursors manually.

## Rate limits

Rate limits are applied per site:

- Growth plans: 600 requests per minute.
- Enterprise plans: 3,000 requests per minute.
- Starter plans: 120 requests per minute.

When a request is throttled the API returns HTTP **429 Too Many Requests**. The
response includes a `Retry-After` header in seconds plus `X-RateLimit-Limit`,
`X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers. Retry after the
indicated delay; clients that retry immediately may be blocked for five minutes.

## Errors and idempotency

Errors use a JSON body with `code`, `message`, and `request_id`. Include the
`request_id` when contacting support. Mutating requests accept an
`Idempotency-Key` header; keys are remembered for 24 hours.

## Authentication details

A missing or expired token returns HTTP 401 with code `AUTH-401`. Insufficient
permissions return HTTP 403 with code `AUTH-403`. Rotate tokens in the console;
an old token remains valid for a 24-hour overlap window after rotation.
