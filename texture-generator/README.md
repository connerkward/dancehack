# Texture generator

3D path viewer and NLE-style timeline for annotating dancer sensor paths with semantic tags and keyframed texture fade. Texture generation (tileable, from text prompt + IP-Adapter) will be provided by the local Python service in `generator/`.

## Frontend

- **Stack:** React, Vite, TypeScript, Three.js via `@react-three/fiber` and `@react-three/drei`.
- **Data:** Mock paths (3D points with velocity), segments (time ranges + tag), tags, and keyframes (fade value over time).

### Run

```bash
cd texture-generator/frontend
npm install
npm run dev
```

Open the URL shown (e.g. http://localhost:5173).

### Layout

- **Viewport:** 3D scene with one tube per path; tube radius varies by velocity.
- **Timeline:** One track per path; segments show tag labels; keyframe dots show fade keyframes.

## Generator (Python)

Planned: FastAPI service with `POST /generate` (text prompt + IP-Adapter image → tileable texture). Not implemented yet.

## Input

The app expects **path data** (ordered 3D points with time), not a raw 3D mesh. If you only have a mesh, you need a separate step to derive paths and choose which end is “start”; see HANDOFF.md §5 for details.

## Data formats

- **Path:** `{ id, name, points: [{ t, x, y, z, velocity? }] }`
- **Segment:** `{ id, trackId, start, end, tagId }`
- **Tag:** `{ id, label, textureId?, textureUrl? }`
- **Keyframe:** `{ id, trackId, time, value }` (value 0–1 for fade)
