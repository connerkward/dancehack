import { useMemo, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { Path, Segment, Keyframe, Tag } from '../types';

const DEFAULT_COLOR = new THREE.Color('#6366f1');
const MAX_DISPLACEMENT = 0.025;
const TUBE_SEGS = 200;
const RAD_SEGS = 16;

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

/* ─── Texture loading hook ─────────────────────────────────────────── */

interface TagTextureEntry {
  map: THREE.Texture | null;
  dispMap: THREE.Texture | null;
}

function useTagTextures(tags: Tag[]): Map<string, TagTextureEntry> {
  const textureKey = tags
    .map((t) => `${t.id}:${t.textureUrl ?? ''}:${t.displacementUrl ?? ''}`)
    .join('|');

  const [textures, setTextures] = useState<Map<string, TagTextureEntry>>(
    new Map(),
  );

  useEffect(() => {
    const loader = new THREE.TextureLoader();
    const loaded: THREE.Texture[] = [];
    const newMap = new Map<string, TagTextureEntry>();

    for (const tag of tags) {
      const entry: TagTextureEntry = { map: null, dispMap: null };
      if (tag.textureUrl) {
        const tex = loader.load(tag.textureUrl);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.repeat.set(10, 3);
        tex.colorSpace = THREE.SRGBColorSpace;
        entry.map = tex;
        loaded.push(tex);
      }
      if (tag.displacementUrl) {
        const tex = loader.load(tag.displacementUrl);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.repeat.set(10, 3);
        entry.dispMap = tex;
        loaded.push(tex);
      }
      if (entry.map || entry.dispMap) {
        newMap.set(tag.id, entry);
      }
    }

    setTextures(newMap);
    return () => {
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
}: {
  path: Path;
  segments: Segment[];
  keyframes: Keyframe[];
  tags: Tag[];
  tagTextures: Map<string, TagTextureEntry>;
  duration: number;
}) {
  const { curve, radiusAt } = useMemo(() => {
    const points = path.points.map((p) => new THREE.Vector3(p.x, p.y, p.z));
    const curve = new THREE.CatmullRomCurve3(points);
    const maxV = Math.max(...path.points.map((p) => p.velocity ?? 1));
    const minV = Math.min(...path.points.map((p) => p.velocity ?? 0.3));
    const radiusAt = (t: number) => {
      const i = Math.min(
        Math.floor(t * (path.points.length - 1)),
        path.points.length - 2,
      );
      const v = path.points[i]?.velocity ?? 0.5;
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

    // Build vertices
    for (let i = 0; i <= TUBE_SEGS; i++) {
      const t = i / TUBE_SEGS;
      const timeAtT = t * duration;
      const r = radiusAt(t);
      const pos = curve.getPoint(t);
      const tangent = curve.getTangent(t);
      const up = new THREE.Vector3(0, 1, 0);
      const normal = new THREE.Vector3()
        .crossVectors(tangent, up)
        .normalize();
      if (normal.lengthSq() < 0.01) {
        up.set(1, 0, 0);
        normal.crossVectors(tangent, up).normalize();
      }
      const binormal = new THREE.Vector3().crossVectors(tangent, normal);

      const seg = getSegmentAtTime(trackSegs, timeAtT);
      const tagEntry = seg ? tagById.get(seg.tagId) : undefined;
      const baseColor = tagEntry
        ? new THREE.Color(tagEntry.color)
        : DEFAULT_COLOR;
      const fade = interpolateKeyframes(trackKfs, timeAtT);

      for (let j = 0; j <= RAD_SEGS; j++) {
        const a = (j / RAD_SEGS) * Math.PI * 2;
        const nx = normal.x * Math.cos(a) + binormal.x * Math.sin(a);
        const ny = normal.y * Math.cos(a) + binormal.y * Math.sin(a);
        const nz = normal.z * Math.cos(a) + binormal.z * Math.sin(a);
        positions.push(pos.x + nx * r, pos.y + ny * r, pos.z + nz * r);
        uvs.push(t, j / RAD_SEGS);
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
        indices.push(a, b, c, c, b, d);
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

      if (texEntry?.map) {
        // Textured material — displacement scale from fade at segment midpoint
        const midTime = seg ? (seg.start + seg.end) / 2 : 0;
        const fadeVal = interpolateKeyframes(trackKfs, midTime);
        matKey = `tex:${tagId}:${fadeVal.toFixed(2)}`;

        if (!matKeyToIndex.has(matKey)) {
          matKeyToIndex.set(matKey, mats.length);
          mats.push(
            new THREE.MeshStandardMaterial({
              map: texEntry.map,
              displacementMap: texEntry.dispMap ?? undefined,
              displacementScale: fadeVal * MAX_DISPLACEMENT,
              metalness: 0.35,
              roughness: 0.45,
            }),
          );
        }
      } else {
        // Default vertex-color material
        matKey = 'default';
        if (!matKeyToIndex.has(matKey)) {
          matKeyToIndex.set(matKey, mats.length);
          mats.push(
            new THREE.MeshStandardMaterial({
              vertexColors: true,
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
  }, [curve, radiusAt, trackSegs, trackKfs, tagById, tagTextures, duration]);

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

/* ─── Scene ────────────────────────────────────────────────────────── */

function Scene({
  paths,
  segments,
  keyframes,
  tags,
  currentTime,
  duration,
}: {
  paths: Path[];
  segments: Segment[];
  keyframes: Keyframe[];
  tags: Tag[];
  currentTime: number;
  duration: number;
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
          />
          <PlayheadMarker
            path={path}
            currentTime={currentTime}
            duration={duration}
          />
        </group>
      ))}
      <OrbitControls makeDefault />
    </>
  );
}

/* ─── PathView ─────────────────────────────────────────────────────── */

export default function PathView({
  paths,
  segments = [],
  keyframes = [],
  tags = [],
  currentTime = 0,
  duration = 8,
}: {
  paths: Path[];
  segments?: Segment[];
  keyframes?: Keyframe[];
  tags?: Tag[];
  currentTime?: number;
  duration?: number;
}) {
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
        />
      </Canvas>
    </div>
  );
}
