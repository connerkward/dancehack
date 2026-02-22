import { useMemo, useState, useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { USDZExporter } from 'three/examples/jsm/exporters/USDZExporter.js';
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js';
import type { Path, Segment, Keyframe, Tag } from '../types';

const TUBE_SEGS = 400;
const RAD_SEGS = 64;

function getSegmentAtTime(segments: Segment[], time: number): Segment | null {
  for (const seg of segments) {
    if (time >= seg.start && time <= seg.end) return seg;
  }
  return null;
}

function interpolateKeyframes(keyframes: Keyframe[], time: number): number {
  if (keyframes.length === 0) return 1;
  const sorted = [...keyframes].sort((a, b) => a.time - b.time);
  if (time <= sorted[0].time) return sorted[0].value;
  if (time >= sorted[sorted.length - 1].time) return sorted[sorted.length - 1].value;
  for (let i = 0; i < sorted.length - 1; i++) {
    if (time >= sorted[i].time && time <= sorted[i + 1].time) {
      const t = (time - sorted[i].time) / (sorted[i + 1].time - sorted[i].time);
      return sorted[i].value + t * (sorted[i + 1].value - sorted[i].value);
    }
  }
  return 1;
}

/* ─── Displacement pixel data (CPU-side) ──────────────────────────── */

interface DispPixels {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

function loadImagePixels(url: string): Promise<DispPixels> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      const id = ctx.getImageData(0, 0, img.width, img.height);
      resolve({ data: id.data, width: img.width, height: img.height });
    };
    img.onerror = reject;
    img.crossOrigin = 'anonymous';
    img.src = url;
  });
}

/** Sample displacement 0-1 from pixel data at wrapped UV. */
function sampleDisp(px: DispPixels, u: number, v: number): number {
  const wu = ((u % 1) + 1) % 1;
  const wv = ((v % 1) + 1) % 1;
  const x = Math.min(Math.floor(wu * px.width), px.width - 1);
  const y = Math.min(Math.floor(wv * px.height), px.height - 1);
  const idx = (y * px.width + x) * 4;
  // Luminance from RGB
  return (px.data[idx] * 0.299 + px.data[idx + 1] * 0.587 + px.data[idx + 2] * 0.114) / 255;
}

/* ─── Texture loading hook ─────────────────────────────────────────── */

interface TagTextureEntry {
  map: THREE.Texture | null;
  normalMap: THREE.Texture | null;
  dispPixels: DispPixels | null;
}

function useTagTextures(tags: Tag[]): Map<string, TagTextureEntry> {
  const textureKey = tags
    .map((t) => `${t.id}:${t.textureUrl ?? ''}:${t.displacementUrl ?? ''}:${t.normalUrl ?? ''}`)
    .join('|');

  const [textures, setTextures] = useState<Map<string, TagTextureEntry>>(
    new Map(),
  );

  useEffect(() => {
    let cancelled = false;
    const loader = new THREE.TextureLoader();
    const loaded: THREE.Texture[] = [];

    (async () => {
      const newMap = new Map<string, TagTextureEntry>();

      for (const tag of tags) {
        const entry: TagTextureEntry = { map: null, normalMap: null, dispPixels: null };
        if (tag.textureUrl) {
          const tex = loader.load(tag.textureUrl);
          tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
          tex.colorSpace = THREE.SRGBColorSpace;
          entry.map = tex;
          loaded.push(tex);
        }
        if (tag.normalUrl) {
          const tex = loader.load(tag.normalUrl);
          tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
          entry.normalMap = tex;
          loaded.push(tex);
        }
        // Load displacement pixel data for CPU-side vertex displacement
        const dispUrl = tag.displacementUrl || tag.textureUrl;
        if (dispUrl) {
          try {
            entry.dispPixels = await loadImagePixels(dispUrl);
          } catch (e) {
            console.warn('Failed to load displacement pixels for', tag.id, e);
          }
        }
        if (entry.map || entry.dispPixels || entry.normalMap) {
          newMap.set(tag.id, entry);
        }
      }

      if (!cancelled) {
        setTextures(newMap);
      }
    })();

    return () => {
      cancelled = true;
      loaded.forEach((t) => t.dispose());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [textureKey]);

  return textures;
}

/* ─── PathTube with multi-material texture + displacement ──────────── */

function PathTube({
  path,
  segments,
  keyframes,
  tags,
  tagTextures,
  duration,
  maxDisplacement = 1,
  showTextures = true,
  textureDensity = 1,
}: {
  path: Path;
  segments: Segment[];
  keyframes: Keyframe[];
  tags: Tag[];
  tagTextures: Map<string, TagTextureEntry>;
  duration: number;
  maxDisplacement?: number;
  showTextures?: boolean;
  textureDensity?: number;
}) {
  const { curve, radiusAt } = useMemo(() => {
    const points = path.points.map((p) => new THREE.Vector3(p.x, p.y, p.z));
    const curve = new THREE.CatmullRomCurve3(points);
    const maxV = Math.max(...path.points.map((p) => p.velocity ?? 1));
    const minV = Math.min(...path.points.map((p) => p.velocity ?? 0.3));
    const radiusAt = (t: number) => {
      const idx = t * (path.points.length - 1);
      const i = Math.min(Math.floor(idx), path.points.length - 2);
      const frac = idx - i;
      const v0 = path.points[i]?.velocity ?? 0.5;
      const v1 = path.points[Math.min(i + 1, path.points.length - 1)]?.velocity ?? 0.5;
      const v = v0 + frac * (v1 - v0);
      return 0.04 + 0.06 * ((v - minV) / (maxV - minV || 1));
    };
    return { curve, radiusAt };
  }, [path]);

  const tagById = useMemo(() => {
    const m = new Map<string, Tag>();
    tags.forEach((t) => m.set(t.id, t));
    return m;
  }, [tags]);

  const trackSegs = useMemo(
    () => segments.filter((s) => s.trackId === path.id),
    [segments, path.id],
  );
  const trackKfs = useMemo(
    () =>
      [...keyframes.filter((k) => k.trackId === path.id)].sort(
        (a, b) => a.time - b.time,
      ),
    [keyframes, path.id],
  );

  const { geometry, materials } = useMemo(() => {
    const positions: number[] = [];
    const uvs: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];

    // Pre-compute arc-length parameterized UVs for uniform texture density.
    // First pass: accumulate chord-length distances between successive samples.
    const arcLengths = new Float64Array(TUBE_SEGS + 1);
    const radii = new Float64Array(TUBE_SEGS + 1);
    const curvePoints: THREE.Vector3[] = [];
    arcLengths[0] = 0;
    for (let i = 0; i <= TUBE_SEGS; i++) {
      const t = i / TUBE_SEGS;
      curvePoints.push(curve.getPoint(t));
      radii[i] = radiusAt(t);
      if (i > 0) {
        arcLengths[i] = arcLengths[i - 1] + curvePoints[i].distanceTo(curvePoints[i - 1]);
      }
    }
    const totalArc = arcLengths[TUBE_SEGS];
    // Average circumference for determining how many times texture tiles around
    let sumRadius = 0;
    for (let i = 0; i <= TUBE_SEGS; i++) sumRadius += radii[i];
    const avgRadius = sumRadius / (TUBE_SEGS + 1);
    const avgCircumference = 2 * Math.PI * avgRadius;
    // Texture tiles per unit length along the tube (scaled down for larger tiles)
    const tilesAlongTube = (totalArc / avgCircumference) * textureDensity;

    // Compute smooth reference frames via parallel transport (eliminates twist ridges)
    const frames = curve.computeFrenetFrames(TUBE_SEGS, false);

    // Build vertices
    for (let i = 0; i <= TUBE_SEGS; i++) {
      const t = i / TUBE_SEGS;
      const r = radii[i];
      const pos = curvePoints[i];
      const normal = frames.normals[i];
      const binormal = frames.binormals[i];

      // Arc-length based U coordinate: how far along the tube in "tile units"
      const u = totalArc > 0 ? (arcLengths[i] / totalArc) * tilesAlongTube : 0;

      const timeAtT = t * duration;
      const seg = getSegmentAtTime(trackSegs, timeAtT);
      const tagId = seg?.tagId;
      const tagEntry = tagId ? tagById.get(tagId) : undefined;
      const baseColor = tagEntry
        ? new THREE.Color(tagEntry.color)
        : new THREE.Color('#ffffff');
      const fade = interpolateKeyframes(trackKfs, timeAtT);

      // CPU displacement: look up pixel data for this tag
      const texEntry = tagId ? tagTextures.get(tagId) : undefined;
      const dispPixels = texEntry?.dispPixels ?? null;
      const tagDispScale = tagEntry?.displacementScale ?? 0.5;
      const dispScale = dispPixels ? fade * tagDispScale * maxDisplacement : 0;

      for (let j = 0; j <= RAD_SEGS; j++) {
        const a = (j / RAD_SEGS) * Math.PI * 2;
        const nx = normal.x * Math.cos(a) + binormal.x * Math.sin(a);
        const ny = normal.y * Math.cos(a) + binormal.y * Math.sin(a);
        const nz = normal.z * Math.cos(a) + binormal.z * Math.sin(a);

        let px = pos.x + nx * r;
        let py = pos.y + ny * r;
        let pz = pos.z + nz * r;

        // Displace vertex along radial normal based on texture
        if (dispPixels && dispScale > 0) {
          const v = j / RAD_SEGS;
          const d = sampleDisp(dispPixels, u, v);
          const offset = (d - 0.5) * dispScale;
          px += nx * offset;
          py += ny * offset;
          pz += nz * offset;
        }

        positions.push(px, py, pz);
        uvs.push(u, j / RAD_SEGS);
        colors.push(
          baseColor.r * fade,
          baseColor.g * fade,
          baseColor.b * fade,
        );
      }
    }

    // Build indices
    for (let i = 0; i < TUBE_SEGS; i++) {
      for (let j = 0; j < RAD_SEGS; j++) {
        const a = i * (RAD_SEGS + 1) + j;
        const b = a + RAD_SEGS + 1;
        const c = a + 1;
        const d = b + 1;
        indices.push(a, c, b, c, d, b);
      }
    }

    // Determine which data segment owns each tubular segment
    const tubeSegOwner: (string | null)[] = [];
    for (let i = 0; i < TUBE_SEGS; i++) {
      const t = (i + 0.5) / TUBE_SEGS;
      const seg = getSegmentAtTime(trackSegs, t * duration);
      tubeSegOwner.push(seg?.id ?? null);
    }

    // Group consecutive tubular segments with the same owner
    interface Run {
      segId: string | null;
      iStart: number;
      iEnd: number;
    }
    const runs: Run[] = [];
    let runStart = 0;
    let runId = tubeSegOwner[0] ?? null;
    for (let i = 1; i < TUBE_SEGS; i++) {
      const id = tubeSegOwner[i] ?? null;
      if (id !== runId) {
        runs.push({ segId: runId, iStart: runStart, iEnd: i });
        runId = id;
        runStart = i;
      }
    }
    runs.push({ segId: runId, iStart: runStart, iEnd: TUBE_SEGS });

    // Build geometry
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(positions, 3),
    );
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    // Build materials array and assign geometry groups
    const mats: THREE.Material[] = [];
    const matKeyToIndex = new Map<string, number>();

    for (const run of runs) {
      const seg = run.segId
        ? trackSegs.find((s) => s.id === run.segId)
        : null;
      const tagId = seg?.tagId;
      const texEntry = tagId ? tagTextures.get(tagId) : undefined;

      let matKey: string;

      if (showTextures && texEntry) {
        // Textured material — color map + normal (displacement is CPU-side in geometry)
        matKey = `tex:${tagId}`;

        if (!matKeyToIndex.has(matKey)) {
          matKeyToIndex.set(matKey, mats.length);
          const tagEntry2 = tagId ? tagById.get(tagId) : undefined;
          const tagDispScale2 = tagEntry2?.displacementScale ?? 0.5;
          const mat = new THREE.MeshStandardMaterial({
            map: texEntry.map ?? undefined,
            metalness: 0.35,
            roughness: 0.45,
          });
          if (texEntry.normalMap) {
            mat.normalMap = texEntry.normalMap;
            mat.normalScale = new THREE.Vector2(tagDispScale2 * maxDisplacement * 2, tagDispScale2 * maxDisplacement * 2);
          }
          mats.push(mat);
        }
      } else {
        // White material (no textures, displacement visible via geometry)
        matKey = 'default';
        if (!matKeyToIndex.has(matKey)) {
          matKeyToIndex.set(matKey, mats.length);
          mats.push(
            new THREE.MeshStandardMaterial({
              color: 0xffffff,
              metalness: 0.3,
              roughness: 0.5,
            }),
          );
        }
      }

      const start = run.iStart * RAD_SEGS * 6;
      const count = (run.iEnd - run.iStart) * RAD_SEGS * 6;
      geo.addGroup(start, count, matKeyToIndex.get(matKey)!);
    }

    return { geometry: geo, materials: mats };
  }, [curve, radiusAt, trackSegs, trackKfs, tagById, tagTextures, duration, maxDisplacement, showTextures, textureDensity]);

  // Dispose old geometry + materials on rebuild
  useEffect(() => {
    return () => {
      geometry.dispose();
      materials.forEach((m) => m.dispose());
    };
  }, [geometry, materials]);

  return <mesh geometry={geometry} material={materials} />;
}

/* ─── Playhead marker ──────────────────────────────────────────────── */

function PlayheadMarker({
  path,
  currentTime,
  duration,
}: {
  path: Path;
  currentTime: number;
  duration: number;
}) {
  const position = useMemo(() => {
    const t = Math.max(0, Math.min(1, currentTime / duration));
    const points = path.points.map((p) => new THREE.Vector3(p.x, p.y, p.z));
    const curve = new THREE.CatmullRomCurve3(points);
    return curve.getPoint(t);
  }, [path, currentTime, duration]);

  return (
    <mesh position={[position.x, position.y, position.z]}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshStandardMaterial
        color="#f59e0b"
        emissive="#f59e0b"
        emissiveIntensity={0.5}
      />
    </mesh>
  );
}

/* ─── Scene exporter helper ────────────────────────────────────────── */

interface ExportFns {
  glb: (() => void) | null;
  usdz: (() => void) | null;
  obj: (() => void) | null;
}

function SceneExporter({
  exportFnRef,
}: {
  exportFnRef: React.MutableRefObject<ExportFns>;
}) {
  const { scene } = useThree();

  useEffect(() => {
    /** Collect mesh objects for export (skip lights, helpers, playhead spheres). */
    const buildExportScene = () => {
      const exportScene = new THREE.Scene();
      scene.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (mesh.isMesh && mesh.geometry) {
          const geo = mesh.geometry as THREE.SphereGeometry;
          if (geo.type === 'SphereGeometry') return;
          exportScene.add(mesh.clone());
        }
      });
      return exportScene;
    };

    exportFnRef.current.glb = () => {
      const exportScene = buildExportScene();
      const exporter = new GLTFExporter();
      exporter.parse(
        exportScene,
        (result) => {
          const blob = new Blob(
            [result as ArrayBuffer],
            { type: 'application/octet-stream' },
          );
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `dancehack-mesh-${Date.now()}.glb`;
          a.click();
          URL.revokeObjectURL(url);
        },
        (error) => {
          console.error('GLTFExporter error:', error);
        },
        { binary: true },
      );
    };

    exportFnRef.current.usdz = async () => {
      const exportScene = buildExportScene();
      const exporter = new USDZExporter();
      try {
        // Runtime method is `parse` (async) despite @types declaring `parseAsync`
        const arraybuffer = await (exporter as any).parse(exportScene);
        const blob = new Blob([arraybuffer], { type: 'model/vnd.usdz+zip' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dancehack-mesh-${Date.now()}.usdz`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (error) {
        console.error('USDZExporter error:', error);
      }
    };

    exportFnRef.current.obj = () => {
      const exportScene = buildExportScene();
      const exporter = new OBJExporter();
      const result = exporter.parse(exportScene);
      const blob = new Blob([result], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dancehack-mesh-${Date.now()}.obj`;
      a.click();
      URL.revokeObjectURL(url);
    };

    return () => {
      exportFnRef.current = { glb: null, usdz: null, obj: null };
    };
  }, [scene, exportFnRef]);

  return null;
}

/* ─── Scene ────────────────────────────────────────────────────────── */

function Scene({
  paths,
  segments,
  keyframes,
  tags,
  currentTime,
  duration,
  maxDisplacement,
  showTextures,
  textureDensity,
  exportFnRef,
}: {
  paths: Path[];
  segments: Segment[];
  keyframes: Keyframe[];
  tags: Tag[];
  currentTime: number;
  duration: number;
  maxDisplacement: number;
  showTextures: boolean;
  textureDensity: number;
  exportFnRef: React.MutableRefObject<ExportFns>;
}) {
  const tagTextures = useTagTextures(tags);

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[4, 6, 4]} intensity={1} />
      <directionalLight position={[-2, 4, -2]} intensity={0.4} />
      {paths.map((path) => (
        <group key={path.id}>
          <PathTube
            path={path}
            segments={segments}
            keyframes={keyframes}
            tags={tags}
            tagTextures={tagTextures}
            duration={duration}
            maxDisplacement={maxDisplacement}
            showTextures={showTextures}
            textureDensity={textureDensity}
          />
          <PlayheadMarker
            path={path}
            currentTime={currentTime}
            duration={duration}
          />
        </group>
      ))}
      <OrbitControls makeDefault />
      <SceneExporter exportFnRef={exportFnRef} />
    </>
  );
}

/* ─── PathView ─────────────────────────────────────────────────────── */

export interface PathViewHandle {
  exportMesh: () => void;
  exportUSDZ: () => void;
  exportOBJ: () => void;
}

const PathView = forwardRef<PathViewHandle, {
  paths: Path[];
  segments?: Segment[];
  keyframes?: Keyframe[];
  tags?: Tag[];
  currentTime?: number;
  duration?: number;
  maxDisplacement?: number;
  showTextures?: boolean;
  textureDensity?: number;
}>(function PathView({
  paths,
  segments = [],
  keyframes = [],
  tags = [],
  currentTime = 0,
  duration = 8,
  maxDisplacement = 1,
  showTextures = true,
  textureDensity = 1,
}, ref) {
  const exportFnRef = useRef<ExportFns>({ glb: null, usdz: null, obj: null });

  useImperativeHandle(ref, () => ({
    exportMesh: () => {
      if (exportFnRef.current.glb) {
        exportFnRef.current.glb();
      } else {
        console.warn('Scene not ready for export');
      }
    },
    exportUSDZ: () => {
      if (exportFnRef.current.usdz) {
        exportFnRef.current.usdz();
      } else {
        console.warn('Scene not ready for export');
      }
    },
    exportOBJ: () => {
      if (exportFnRef.current.obj) {
        exportFnRef.current.obj();
      } else {
        console.warn('Scene not ready for export');
      }
    },
  }));

  return (
    <div className="viewport">
      <Canvas camera={{ position: [3, 2, 3], fov: 50 }}>
        <Scene
          paths={paths}
          segments={segments}
          keyframes={keyframes}
          tags={tags}
          currentTime={currentTime}
          duration={duration}
          maxDisplacement={maxDisplacement}
          showTextures={showTextures}
          textureDensity={textureDensity}
          exportFnRef={exportFnRef}
        />
      </Canvas>
    </div>
  );
});

export default PathView;
