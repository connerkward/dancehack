# Texture generator — handoff for Claude

This document describes the **texture-generator** subfolder so another Claude session (or a human) can continue work without re-discovering the codebase.

---

## 1. Purpose

- **Input:** Dancer sensor data → 3D paths where each path has points `(t, x, y, z)` and optional **velocity** (used as tube width).
- **Goal:** Add textures onto these curves. The user annotates **segments** on an NLE-style **timeline** with **semantic tags**; each tag maps to a texture. Textures can be **keyframed** to fade (e.g. displacement or opacity 0→1) over time.
- **Texture generation (planned):** Local Python service. Input: **text prompt** + **IP-Adapter** reference image. Output: **tileable** texture (arbitrarily long along the track). Inpaint-based seam fixing is the intended approach.

---

## 2. Repo context

- **Repo:** `dancehack` (root has only a README; no app code at root).
- **Scope:** All feature work lives under `texture-generator/`. The rest of the repo is effectively empty.

---

## 3. Folder structure

```
texture-generator/
├── frontend/                 # Web app (React + Vite + R3F)
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       ├── types/
│       │   └── index.ts       # Path, Segment, Tag, Keyframe, etc.
│       ├── data/
│       │   └── mockData.ts    # Mock paths, segments, tags, keyframes
│       └── components/
│           ├── PathView.tsx   # 3D tubes from paths (velocity = width)
│           └── Timeline.tsx   # NLE track rows + segments + keyframe dots
├── generator/                # (Planned) Local Python texture service — not implemented
├── README.md                 # Run instructions + data format summary
└── HANDOFF.md                # This file
```

---

## 4. Data model (TypeScript interfaces)

**File:** `frontend/src/types/index.ts`

| Type | Purpose |
|------|--------|
| **PathPoint** | One sample: `t` (seconds), `x, y, z`, optional `velocity`. Width of the 3D tube is derived from velocity. |
| **Path** | One path = one timeline track: `id`, `name`, `points: PathPoint[]`. |
| **Segment** | Annotated time range on a track: `id`, `trackId`, `start`, `end`, `tagId`. |
| **Tag** | Semantic label → texture: `id`, `label`, optional `textureId`, `textureUrl` (filled when generator produces a texture). |
| **Keyframe** | Fade control: `id`, `trackId`, `time` (seconds), `value` (0–1, e.g. opacity or displacement). |
| **Track** | Convenience: `path` + `segments` + `keyframes` for one track. |
| **AppState** | Full state: `paths`, `tags`, `segments`, `keyframes`. |

---

## 5. Input: paths only; using a 3D model (mesh)

The app does **not** accept a raw 3D mesh as input. It expects **path data**: ordered lists of 3D points with time (`PathPoint[]`). The curve is built from array order; the first point is the path “start” (time 0), the last is the “end” (time = duration). Segments and keyframes are defined in seconds along that timeline.

**If you only have a 3D model (mesh):** the current software will not work as-is. To support a mesh-only workflow you would add a **mesh → paths** step that:

1. **Extracts or lets the user define one or more paths on the mesh**, e.g.:
   - User draws a path on the surface, or
   - User picks start/end vertices and the app computes a path (e.g. along edges or a centerline), or
   - Extract edge loops or a skeleton and treat them as paths.
2. **Outputs `Path` data:** ordered 3D points plus synthetic `t` (e.g. `t = (index / (n-1)) * duration`) and optional `velocity`.
3. **Lets the user choose which side is “start”:** e.g. a “Reverse path” control or “Set start here” so that the first point in the array (and thus time 0) is the end they want. That choice only affects point order.

Once paths exist in the current format, the rest of the pipeline (PathView, timeline, segments, keyframes, generator) works unchanged.

---

## 6. Mock data

**File:** `frontend/src/data/mockData.ts`

- **Paths:** Two paths: "Right hand" (80 points, 8s), "Left hand" (60 points, 8s). 3D curves built from parametric sin/cos; `velocity` varies per point.
- **Tags:** Four: Fire, Water, Organic, Metallic (ids `tag-fire`, etc.). No `textureUrl` yet.
- **Segments:** Five segments across the two tracks; each has a `tagId` and time range (e.g. 0–2.5s Fire on right hand, 2.5–5s Water, etc.).
- **Keyframes:** Seven keyframes (e.g. right-hand fade 0→1→1→0 over 8s; left-hand different curve).

All exports: `mockPaths`, `mockTags`, `mockSegments`, `mockKeyframes`.

---

## 7. UI components

### 7.1 PathView (`frontend/src/components/PathView.tsx`)

- **Stack:** `@react-three/fiber`, `@react-three/drei`, `three`.
- **Behavior:** Renders a **tube** per path. Curve from `PathPoint[]` via `CatmullRomCurve3`; tube radius at each point is derived from `velocity` (min/max normalized to a radius range). Single material (e.g. purple) for now; no per-segment textures yet.
- **Props:** `paths: Path[]`.
- **Layout:** Fills the `.viewport` div (flex child). Canvas has orbit controls.

### 7.2 Timeline (`frontend/src/components/Timeline.tsx`)

- **Behavior:** NLE-style panel. Time ruler (0 to `duration` seconds) with ticks; one **track row** per path. Each row shows:
  - **Segments:** Colored blocks by `tagId` (fixed color map for the four mock tags), positioned by `start`/`end` in pixels (`PIXELS_PER_SECOND = 80`).
  - **Keyframes:** Small dots at `time` with tooltip (time + value).
- **Props:** `paths`, `segments`, `tags`, `keyframes`, `duration` (default 8).

### 7.3 App (`frontend/src/App.tsx`)

- **Layout:** Column: top = viewport (PathView), bottom = Timeline. Uses mock data only; no state updates or persistence yet.

---

## 8. Styling

**File:** `frontend/src/index.css`

- `.app`: column, full height.
- `.viewport`: flex grow, min height ~280px, dark background (3D scene).
- `.timeline-panel`: fixed height ~240px, dark panel, border-top.
- `.track-row`, `.track-label`, `.track-lane`, `.segment`, `.keyframe-dot`, `.timeline-ruler`, etc.: layout and colors for the timeline (dark theme).

---

## 9. How to run

```bash
cd texture-generator/frontend
npm install
npm run dev
```

Then open the dev URL (e.g. http://localhost:5173). Build: `npm run build` (output in `dist/`).

---

## 10. Planned / not yet implemented

1. **Dancer video window**  
   Add a panel (e.g. left of the 3D viewport) that shows video(s) of the dancers. Use `<video>` with a mock or real URL; optional: sync `currentTime` with timeline time. Suggested: new component `DancerVideoWindow` (or `VideoPanel`), then in `App` use a horizontal split (video panel | PathView) above the timeline.

2. **Generator service (Python)**  
   FastAPI app under `generator/`. Endpoint e.g. `POST /generate`: body = text prompt + IP-Adapter image (file or base64) + optional length/tile count; response = tileable texture image(s). Local pipeline: ComfyUI or diffusers + IP-Adapter + inpaint for seamless horizontal tiling.

3. **Texture mapping on path**  
   Map segment time ranges to UVs along each tube; assign texture per tag (from generator or placeholder); apply keyframe-driven fade (opacity or displacement) in material/shader.

4. **Interactivity**  
   Add/edit/delete segments; assign tags; add/move keyframes; optional playback with timeline scrubbing and video sync.

5. **Persistence**  
   Save/load paths, segments, tags, keyframes (e.g. JSON or backend).

---

## 11. Generator API contract (for when backend exists)

- **Request:** e.g. `POST /generate` with `{ prompt: string, ipAdapterImage: string (base64 or URL), segmentLength?: number, tag?: string }`.
- **Response:** Image (PNG) or tiles + tiling hint so the frontend can repeat the texture along the path.
- **Health:** `GET /health` optional.

---

## 12. Quick reference: key files

| What | Where |
|------|--------|
| Types | `frontend/src/types/index.ts` |
| Mock data | `frontend/src/data/mockData.ts` |
| 3D path tubes | `frontend/src/components/PathView.tsx` |
| NLE timeline | `frontend/src/components/Timeline.tsx` |
| App layout + data wiring | `frontend/src/App.tsx` |
| Global styles | `frontend/src/index.css` |
| Run/docs | `texture-generator/README.md` |

Use this file to onboard Claude (or a human) and continue from the current state: add video window, implement generator, wire textures to segments, and add interaction/persistence as needed.
