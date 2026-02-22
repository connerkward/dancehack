import type { Path, PathPoint, Segment, Tag, Keyframe } from '../types';

/** Mock path: smooth 3D curve with velocity (width) variation. */
function makePathPoints(
  count: number,
  duration: number,
  scale: number = 1
): PathPoint[] {
  const points: PathPoint[] = [];
  for (let i = 0; i < count; i++) {
    const t = (i / (count - 1)) * duration;
    const u = (i / (count - 1)) * Math.PI * 2;
    const v = (i / (count - 1)) * Math.PI;
    points.push({
      t,
      x: scale * Math.sin(u) * (1 + 0.3 * Math.cos(v)),
      y: scale * (0.5 * Math.cos(v) + 0.5),
      z: scale * Math.cos(u) * (1 + 0.3 * Math.cos(v)),
      velocity: 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(u * 2)),
    });
  }
  return points;
}

export const mockPaths: Path[] = [
  {
    id: 'path-right-hand',
    name: 'Right hand',
    points: makePathPoints(80, 8, 1.2),
  },
  {
    id: 'path-left-hand',
    name: 'Left hand',
    points: makePathPoints(60, 8, 1.0).map((p) => ({
      ...p,
      x: -p.x,
      z: p.z + 0.5,
    })),
  },
];

export const mockTags: Tag[] = [
  { id: 'tag-fire', label: 'Fire', color: '#ef4444', textureId: null, textureUrl: null },
  { id: 'tag-water', label: 'Water', color: '#3b82f6', textureId: null, textureUrl: null },
  { id: 'tag-organic', label: 'Organic', color: '#22c55e', textureId: null, textureUrl: null },
  { id: 'tag-metallic', label: 'Metallic', color: '#a1a1aa', textureId: null, textureUrl: null },
];

export const mockSegments: Segment[] = [
  { id: 'seg-1', trackId: 'path-right-hand', start: 0, end: 2.5, tagId: 'tag-fire', name: 'Intro blaze', prompt: 'fire particle swirl' },
  { id: 'seg-2', trackId: 'path-right-hand', start: 2.5, end: 5, tagId: 'tag-water', name: 'Flow', prompt: 'flowing water ripple' },
  { id: 'seg-3', trackId: 'path-right-hand', start: 5, end: 8, tagId: 'tag-organic', name: 'Vine growth', prompt: 'organic vine texture' },
  { id: 'seg-4', trackId: 'path-left-hand', start: 0.5, end: 4, tagId: 'tag-metallic', name: 'Chrome', prompt: 'brushed metal surface' },
  { id: 'seg-5', trackId: 'path-left-hand', start: 4, end: 8, tagId: 'tag-fire', name: 'Finale fire', prompt: 'intense fire burst' },
];

export const mockKeyframes: Keyframe[] = [
  { id: 'kf-1', trackId: 'path-right-hand', time: 0, value: 0 },
  { id: 'kf-2', trackId: 'path-right-hand', time: 1, value: 1 },
  { id: 'kf-3', trackId: 'path-right-hand', time: 6, value: 1 },
  { id: 'kf-4', trackId: 'path-right-hand', time: 8, value: 0 },
  { id: 'kf-5', trackId: 'path-left-hand', time: 0, value: 0.2 },
  { id: 'kf-6', trackId: 'path-left-hand', time: 4, value: 1 },
  { id: 'kf-7', trackId: 'path-left-hand', time: 8, value: 0.3 },
];
