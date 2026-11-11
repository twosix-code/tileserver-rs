# integration/demo-fixes — container build context

Handoff note for building the tileserver-rs demo container binary from the
**integration branch** `integration/demo-fixes`.

## Why this branch

`integration/demo-fixes` is the integration branch for the demo. It sits on top
of `integrate/all-three-fixes` (the three upstream PR fixes) and stacks the
STAC + dataset-cache hardening commits the demo needs. It is kept merged up to
`origin/main` (last sync: `7796c90b`) so the demo builds on current upstream,
not on the 2026-10-30 base.

```
75fcea31 fix(stac): canonicalize S3 asset hrefs to one host form
4ca4f6c0 feat(stac): coalesce bbox searches + cap per-tile mosaic fan-out
fd3fb73b test(stac): use 3-band raster for the blank-PNG guard test
086ae692 fix(stac): retry first-page bbox search with backoff; never cache blank tiles
ba4e403a refactor(dataset_cache): scope contains_key to tests; rewrite bounded-cache docs
eb2cd2de fix(dataset_cache): cap pool idle list so a stopped source cannot pin the budget
769adde7 fix(dataset_cache): one-shot fallback with bounded wait so budget never errors a tile
32d790c0 fix(dataset_cache): reclaim partially idle pools instead of only fully idle ones
a58d815f feat(dataset_cache): allow overriding the byte budget via env
...
4e564f6e fix(stac): paginate dynamic bbox-search across `links[rel=next]`
57ea4443 fix(manager): route raster tiles through the global tile cache
955b52d4 perf(cog)+fix(dataset_cache): cache overview datasets, bound the cache by bytes
```

Confirmed by the app's own entrypoint docs (`src/maps/TileserverRS/shared/endpoints.js`):
the demo stack is built from this branch.

## Build constraints

- The container does **NOT** compile. `deploy/tileserver/tileserver-runtime/Dockerfile`
  `COPY`s a **prebuilt** native binary. Build the binary on the host, then refresh
  the runtime build context — that refresh is the whole "prep" step.
- **GDAL floor: 3.8+ is not required here, and must not be introduced.** The
  `[patch.crates-io]` pin in `Cargo.toml` targets
  `AlexanderWillner/gdal#fix/gdal-3.13-compat` (`968cf34`), whose build script
  requires only **GDAL >= 3.4**. Upstream `main` pins georust master
  (`c0a6266`), which requires **>= 3.8** (georust/gdal#716) — do not follow it
  on this line. Both the build host and the deploy runtime are Debian 12 with
  GDAL 3.6.x / `libgdal.so.32`; `tileserver-runtime` is `FROM debian:12-slim`
  and ships `runtime-libs/libgdal.so.32`. GDAL's soname bumps per minor
  release, so a 3.8+ build would demand `libgdal.so.34+` and fail to load at
  container start. The master pin's pre-built 3.13 bindings are unused anyway:
  this crate builds with `features = ["bindgen"]` against the local headers.
  Revisit only if the runtime image moves off Debian 12.
- `feature/identity-raster-cache` is **NOT** part of this line. Not folded in
  unless identity raster-tile caching becomes a requirement.
- Keep the GDAL/VSI tuning ENV in `run-native.sh` in sync with the image ENV.
- Husky hooks fail on this host → commit/push with `--no-verify`.

## Steps

```sh
cd ~/.openclaw/work/gits/Gisat/DynamicServerSideTiles/tileserver-rs
git checkout integration/demo-fixes
./build-native.sh              # preflight + cargo build --release

# refresh the container build context
cp target/release/tileserver-rs \
   ../app-gisat-deckglSandbox/deploy/tileserver/tileserver-runtime/tileserver-rs

cd ../app-gisat-deckglSandbox
docker compose -f deploy/tileserver/docker-compose.yml up -d --build
curl -s http://localhost:8080/health
```
