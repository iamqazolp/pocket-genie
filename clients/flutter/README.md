# Pocket Genie — Minimal Flutter Mobile Client

A lightweight Flutter Android/iOS client (`clients/flutter`) that connects to the headless Pocket Genie `/api/v1` server running either:
- **On-device inside Termux on your Xiaomi 14 Pro** (`http://127.0.0.1:8090`), or
- **Over LAN on your Mac** (`http://192.168.0.101:8090`).

## Features
- **Specialist Routing Bar**: Switch between `Auto (Supervisor)` and direct specialists (`DevOps`, `SQL`, `Knowledge`, `Device`).
- **Live Quantization Telemetry Badge**: Displays active precision tier (`t1` / `t2` / `t3`), mixed-precision profile (`P_fast` / `P_mixed` / `P_accurate`), and effective activation bit-width.
- **Collapsible Execution Trace**: Inspect subagent reasoning and tool observations inline under each message.
- **Inline HITL Approval Cards**: One-tap `Approve` / `Reject` buttons whenever a mutating tool is intercepted by `HumanInTheLoopMiddleware`.
- **Instant Endpoint Switcher**: Tap the network icon in the top-right bar to switch between `http://127.0.0.1:8090` (on-phone Termux) and `http://192.168.0.101:8090` (LAN host).

## Quick Start (Xiaomi 14 Pro)

```bash
cd clients/flutter
# Generate platform runner files if building for the first time without overwriting lib/ or pubspec.yaml:
flutter create --platforms=android .
flutter pub get
flutter run -d <xiaomi_device_id>
```
