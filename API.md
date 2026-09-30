# Vehicle Counting API — Integration Guide

For the web developer integrating the hydrogen station waiting-status site.

## Endpoint

```
GET http://192.168.0.167:8080/vehicle-count
```

The service runs in Docker on a mini PC at the station (Ubuntu Server 26.04, container
`hydrogen-counter`, port published to the host). Docker is an implementation detail — from your side
this is a plain HTTP endpoint, no auth, no headers required.

Health check:

```
GET http://192.168.0.167:8080/health
→ 200 {"status":"ok"}
```

## Response

```json
{
  "cameraId": "cam2",
  "WaitingTime": "00:00",
  "totalVehicles": 2,
  "cars": 1,
  "heavyVehicles": 1
}
```

This shape is **fixed**. Do not expect new fields; if you need something else, ask first.

| Field | Type | Meaning |
|---|---|---|
| `cameraId` | string | Logical camera id. Always `cam2` today. |
| `WaitingTime` | string `MM:SS` | Estimated wait for a vehicle arriving now. `"00:00"` = no wait. Note the capital W — it differs from the other keys. |
| `totalVehicles` | int | All detected vehicles in the monitored regions, **including vehicles currently being served at a charger**. |
| `cars` | int | Cars among `totalVehicles`. |
| `heavyVehicles` | int | Buses and trucks among `totalVehicles`. |

### Important: what `totalVehicles` is not

`totalVehicles` counts every vehicle in the monitored area, chargers included. A bus parked at a
dispenser is being served, not queueing, but it still adds to `totalVehicles`. If the site copy says
"N vehicles waiting", that number can overstate the queue. Discuss with Jonibek before changing how
it is displayed — the field itself will not change.

### How WaitingTime behaves

- The station has **2 charger slots**. A bus or truck occupies **both** (it blocks the lane).
- The countdown starts only when every slot is occupied. With one free slot, `WaitingTime` is `00:00`.
- Charge time estimates: car 7 minutes, bus/truck 30 minutes.
- The value is exponentially smoothed, so after the station empties it takes ~20 seconds to fall to
  `00:00` rather than dropping instantly. Do not treat a slowly falling value as a stale response.

## Status codes

| Code | When | What to do |
|---|---|---|
| 200 | Normal | Use the payload. |
| 503 | Right after startup, before the first frame is analysed | Retry. Show your own "loading" state, not zeros. |
| 404 | Unknown path, or a non-GET method | `HEAD` is currently not handled and returns 404 — use `GET`. |

## Polling

Poll every **2–5 seconds**. The detector processes roughly 1 frame per second, so polling faster
returns the same values and only wastes CPU on a machine that is already busy.

Send `Cache-Control: no-cache` or a cache-busting query parameter; responses carry `Cache-Control: no-store`
but intermediaries can still interfere.

## Failure handling — please read

If the camera stream drops, the service keeps returning the **last known values** indefinitely. There
is currently no staleness marker in the payload. A frozen number looks identical to a real one.

Handle this on your side: keep the timestamp of the last change you observed, and if the values have
not moved for several minutes during operating hours (06:00–20:30), show a degraded state rather than
a confident number. If you would prefer the service to return `503` when its data is stale, say so —
that can be added without changing the JSON shape.

## Network reachability — needs a decision

The mini PC sits on a private LAN (`192.168.0.167`). A site hosted on Vercel **cannot reach it**.
Before integrating, we need to agree how the data crosses that boundary:

1. **Push (recommended)** — the mini PC POSTs this JSON to an endpoint you provide, every few seconds.
   Nothing inbound is opened, and it keeps working if the LAN IP changes. Needs a small change on our side;
   tell us the URL and the auth header you want.
2. **Cloudflare Tunnel** — an agent on the mini PC exposes the endpoint on a public hostname. No router
   configuration, free tier is sufficient.
3. **Router port forwarding + DDNS** — simplest, but puts the mini PC directly on the internet. Not preferred.

Note also that browser JavaScript calling this endpoint cross-origin would need CORS headers, which the
service does not send today. Calling it from your server side avoids that entirely.

## Running your own code on the same mini PC

- Ask Jonibek for your own user account (`/home/<you>`), so files do not mix.
- **Port 8080 is taken.** Pick another, e.g. 3000.
- The counting service must keep running. Do not stop or remove the `hydrogen-counter` container;
  it is set to restart automatically and is what feeds the site.
- Docker is installed if you want to containerise your service too.

## Contact

Questions about the detection logic, the regions, or the wait-time model → Jonibek.
