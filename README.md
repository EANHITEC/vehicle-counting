# Vehicle Counting Backend

C# backend for gas-station vehicle detection, region-based counting, and waiting-time estimation.

This service is intended to be used by a website or API consumer that needs a simple answer to the question:

`If I go to the gas station now, how long will I likely have to wait?`

## Overview

The application:

- loads a fine-tuned ONNX model
- detects `car`, `bus`, and `truck`
- tracks vehicles across frames
- applies charger and waiting-lane region logic
- returns a small JSON payload over HTTP

The current implementation is focused on the `cam2` view and the station-specific charger layout used in this project.

## Project Files

```text
.
├── Program.cs
├── VehicleCountingONNX.csproj
├── README.md
├── .gitignore
└── .github/workflows/dotnet.yml
```

## Requirements

- .NET 8 SDK
- Linux environment with OpenCV runtime support
- ONNX model file: `gas_station_yolo11m.onnx`

## Model File

Expected filename:

```text
gas_station_yolo11m.onnx
```

The application expects the ONNX file to be placed in the repository root, next to `Program.cs`.

Recommended distribution options:

- GitHub Release asset
- Git LFS
- private storage / deployment artifact

If the model is distributed through GitHub Releases, download it and place it in the project root before running the service.

## Build

```bash
dotnet restore VehicleCountingONNX.csproj
dotnet build VehicleCountingONNX.csproj
```

## Quick Start

Run with HLS:

```bash
dotnet run --project VehicleCountingONNX.csproj -- \
  --source "https://cctv.hydrogen-cctv.com/hydrogen_cam/index.m3u8" \
  --camera-id cam2 \
  --view
```

Run with RTSP:

```bash
dotnet run --project VehicleCountingONNX.csproj -- \
  --source "rtsp://user:pass@camera-ip:554/stream" \
  --camera-id cam2 \
  --view
```

## API

Default endpoints:

- `GET /vehicle-count`
- `GET /health`

Example response:

```json
{
  "cameraId": "cam2",
  "WaitingTime": "20:00",
  "totalVehicles": 3,
  "cars": 2,
  "heavyVehicles": 1
}
```

### Response Fields

- `cameraId`: logical camera identifier
- `WaitingTime`: estimated waiting time in `MM:SS.`
- `totalVehicles`: total detected vehicles in relevant regions
- `cars`: detected cars
- `heavyVehicles`: detected buses and trucks

## Current Business Logic

- Detection classes are `car`, `bus`, and `truck.`
- Waiting time is driven by the right-side charger and waiting-lane logic
- Region and queue behavior are tuned for the current station layout
- The implementation is station-specific, not a generic multi-site backend

## CI

GitHub Actions workflow:

```text
.github/workflows/dotnet.yml
```

It runs:

- `dotnet restore`
- `dotnet build`

on pushes and pull requests targeting `main`.
