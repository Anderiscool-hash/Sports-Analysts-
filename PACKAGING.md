# Packaging & selling SportEdge

This explains how to build the desktop `.exe`, gate it behind a license, protect
your source, and the realistic limits of each step.

## TL;DR

- `build_exe.bat` → a plain Windows build (works, but **decompilable**).
- `build_exe.bat obf` → obfuscated with PyArmor + packaged (raises the bar a lot).
- A local `.exe` can **never** be made truly un-reversible. To actually keep your
  edge secret, run the model on a **server** and ship a thin client (see below).

---

## 1. The license gate (already built)

Customers can't run the app without a signed key only you can mint.

```bash
# once - create your signing keypair
python scripts/license_tool.py keygen
# -> paste the PUBLIC key into src/sportedge/licensing.py EMBEDDED_PUBLIC_KEY_B64
#    keep the PRIVATE key secret forever (a password manager, not the repo)

# per customer
python scripts/license_tool.py mint --priv <PRIVATE_B64> --sub "tiktok_buyer_07" --tier pro --days 30
# -> send the printed key to the customer
```

The customer saves the key to `%APPDATA%\SportEdge\license.key` (or sets the
`SPORTEDGE_LICENSE` env var). Expiry is enforced offline, so a 30-day key simply
stops working after 30 days — that's your subscription lever.

> Rotation: if a key leaks, ship a build with a new embedded public key; all old
> keys die at once. Keep tiers/days short for TikTok sales.

## 2. Building the exe

```bat
build_exe.bat        REM plain
build_exe.bat obf    REM obfuscated (recommended before selling)
```

Output is `dist\SportEdge\` — **ship the whole folder**. It contains
`SportEdge.exe` plus `models\` and `config\`. The exe `chdir`s to its own folder
on launch, so those relative paths resolve for the customer.

## 3. Protecting the source — what actually works

| Approach | Effort | Protection | Notes |
|---|---|---|---|
| PyInstaller only | low | weak | `pyinstxtractor` + a decompiler recovers your `.py`. |
| PyInstaller + **PyArmor** (`build_exe.bat obf`) | low | medium | Bytecode is encrypted/wrapped; deters the vast majority. Still not bulletproof. |
| **Server-side model** (thin client) | medium | strong | The model + pick logic never ship. Only way to truly hide the edge. |

### The recommended endgame: server-side picks
Because you want to **sell picks on TikTok**, you don't actually need customers to
run the model — you need them to *receive your picks*. Move the valuable part to a
server you control:

1. You run the live loop / model on your own machine or a cheap VPS.
2. It publishes picks to a small authenticated API (or even a private feed).
3. The shipped `.exe` becomes a **viewer**: it sends the license key, pulls today's
   picks, and displays them. No model, no edge logic in the binary.

The license gate and desktop shell built here are already the right front end for
that — swapping the local loop for an API call is a contained change in
`src/sportedge/app.py`. When you're ready, I can build that client/server split.

## 4. Legal / compliance (read before selling)

Selling sports/betting **picks** for money has real regulatory exposure that
varies by jurisdiction (consumer-protection, gambling, and advertising rules), and
Kalshi is a CFTC-regulated venue with its own terms on automated access and
redistribution. Before you charge anyone or post results on TikTok:

- Add clear disclaimers (not financial advice; past results != future returns).
- Check Kalshi's API/automation terms and your local rules on selling picks.
- Consider consulting a lawyer — this doc is not legal advice.
