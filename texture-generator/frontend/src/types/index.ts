/** Single sample along a 3D path (dancer sensor). */
export interface PathPoint {
  t: number;       // time (seconds)
  x: number;
  y: number;
  z: number;
  velocity?: number;  // optional; width derived from this
}

/** One path = one track (e.g. one body part / sensor). */
export interface Path {
  id: string;
  name: string;
  points: PathPoint[];
}

/** Annotated segment on a track with a semantic tag. */
export interface Segment {
  id: string;
  trackId: string;  // path id
  start: number;    // time in seconds
  end: number;
  tagId: string;
  name?: string;    // user-defined label
  prompt?: string;  // texture generation prompt
}

/** Semantic tag → texture (generator will produce texture for this tag). */
export interface Tag {
  id: string;
  label: string;
  prompt?: string;            // texture generation prompt
  color: string;              // display color (hex)
  textureId?: string | null;  // assigned after generation
  textureUrl?: string | null;
  displacementUrl?: string | null;
  normalUrl?: string | null;
  displacementScale?: number;
  referenceImageUrl?: string | null;
  ipAdapterScale?: number;
}

/** Keyframe for fade (opacity or displacement) along the track. */
export interface Keyframe {
  id: string;
  trackId: string;
  time: number;   // seconds
  value: number;  // 0..1, e.g. opacity or displacement amount
}

/** One track in the NLE = one path + its segments + keyframes. */
export interface Track {
  path: Path;
  segments: Segment[];
  keyframes: Keyframe[];
}

/** Full app state (paths, tags, tracks). */
export interface AppState {
  paths: Path[];
  tags: Tag[];
  segments: Segment[];
  keyframes: Keyframe[];
}
