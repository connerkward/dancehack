import { useMemo, useCallback, useState, useRef, useEffect } from 'react';
import type { Path, Segment, Tag, Keyframe } from '../types';

const PIXELS_PER_SECOND = 80;
const LABEL_WIDTH = 120;
const SEGMENT_HEIGHT = 22;
const FADE_LANE_HEIGHT = 40;
const TRACK_HEIGHT = SEGMENT_HEIGHT + FADE_LANE_HEIGHT + 8;
const MIN_SEGMENT_DURATION = 0.2;

const DEFAULT_TAG_COLOR = '#6b7280';

/** Generate SVG polyline points for the fade envelope. */
function buildFadeEnvelopePoints(
  keyframes: Keyframe[],
  duration: number,
  width: number,
  height: number,
): string {
  if (keyframes.length === 0) return '';
  const sorted = [...keyframes].sort((a, b) => a.time - b.time);
  const points: string[] = [];
  const step = 2;
  for (let px = 0; px <= width; px += step) {
    const t = (px / width) * duration;
    const v = interpolateAtSorted(sorted, t);
    const y = height - v * (height - 2);
    points.push(`${px},${y.toFixed(1)}`);
  }
  return points.join(' ');
}

function interpolateAtSorted(sorted: Keyframe[], time: number): number {
  if (sorted.length === 0) return 1;
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

function buildFadeMaskGradient(keyframes: Keyframe[], duration: number): string {
  if (keyframes.length === 0) return 'none';
  const sorted = [...keyframes].sort((a, b) => a.time - b.time);
  const stops: string[] = [];
  const samples = 40;
  for (let i = 0; i <= samples; i++) {
    const t = (i / samples) * duration;
    const pct = (i / samples) * 100;
    const alpha = interpolateAtSorted(sorted, t);
    stops.push(`rgba(0,0,0,${alpha.toFixed(3)}) ${pct.toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

/* ─── Main Timeline ─────────────────────────────────────────────────── */

type Selection = {
  segments: Set<string>;
  keyframes: Set<string>;
};

const EMPTY_SELECTION: Selection = { segments: new Set(), keyframes: new Set() };

export default function Timeline({
  paths,
  segments,
  tags,
  keyframes,
  duration = 8,
  currentTime = 0,
  onSeek,
  onSegmentsChange,
  onKeyframesChange,
}: {
  paths: Path[];
  segments: Segment[];
  tags: Tag[];
  keyframes: Keyframe[];
  duration?: number;
  currentTime?: number;
  onSeek?: (t: number) => void;
  onSegmentsChange?: (segs: Segment[]) => void;
  onKeyframesChange?: (kfs: Keyframe[]) => void;
}) {
  const [draggingKf, setDraggingKf] = useState<string | null>(null);
  const [resizing, setResizing] = useState<{ segId: string; edge: 'start' | 'end'; originX: number; originTime: number } | null>(null);
  const [selection, setSelection] = useState<Selection>(EMPTY_SELECTION);
  const panelRef = useRef<HTMLDivElement>(null);

  const tagById = useMemo(() => {
    const m = new Map<string, Tag>();
    tags.forEach((t) => m.set(t.id, t));
    return m;
  }, [tags]);

  const segmentsByTrack = useMemo(() => {
    const m = new Map<string, Segment[]>();
    segments.forEach((s) => {
      const list = m.get(s.trackId) ?? [];
      list.push(s);
      m.set(s.trackId, list);
    });
    m.forEach((list) => list.sort((a, b) => a.start - b.start));
    return m;
  }, [segments]);

  const keyframesByTrack = useMemo(() => {
    const m = new Map<string, Keyframe[]>();
    keyframes.forEach((k) => {
      const list = m.get(k.trackId) ?? [];
      list.push(k);
      m.set(k.trackId, list);
    });
    m.forEach((list) => list.sort((a, b) => a.time - b.time));
    return m;
  }, [keyframes]);

  const rulerTicks = useMemo(() => {
    const ticks: { t: number; major: boolean }[] = [];
    for (let t = 0; t <= duration; t += 0.5) {
      ticks.push({ t, major: t % 1 === 0 });
    }
    return ticks;
  }, [duration]);

  const rulerWidth = duration * PIXELS_PER_SECOND;
  const playheadX = LABEL_WIDTH + currentTime * PIXELS_PER_SECOND;

  const hasSelection = selection.segments.size > 0 || selection.keyframes.size > 0;

  /* ── Selection helpers ──────────────────────────── */
  const selectSegment = useCallback((segId: string, additive: boolean) => {
    setSelection((prev) => {
      const next = new Set(additive ? prev.segments : []);
      if (next.has(segId)) next.delete(segId);
      else next.add(segId);
      return { segments: next, keyframes: additive ? prev.keyframes : new Set() };
    });
  }, []);

  const selectKeyframe = useCallback((kfId: string, additive: boolean) => {
    setSelection((prev) => {
      const next = new Set(additive ? prev.keyframes : []);
      if (next.has(kfId)) next.delete(kfId);
      else next.add(kfId);
      return { segments: additive ? prev.segments : new Set(), keyframes: next };
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelection(EMPTY_SELECTION);
  }, []);

  /* ── Delete selected items via Delete / Backspace ── */
  const deleteSelection = useCallback(() => {
    if (selection.segments.size > 0 && onSegmentsChange) {
      onSegmentsChange(segments.filter((s) => !selection.segments.has(s.id)));
    }
    if (selection.keyframes.size > 0 && onKeyframesChange) {
      onKeyframesChange(keyframes.filter((k) => !selection.keyframes.has(k.id)));
    }
    setSelection(EMPTY_SELECTION);
  }, [selection, segments, keyframes, onSegmentsChange, onKeyframesChange]);

  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      if (!hasSelection) return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelection();
      }
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [hasSelection, deleteSelection]);

  /* ── Ruler click → seek ─────────────────────────── */
  const handleRulerClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!onSeek) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left - LABEL_WIDTH;
      const t = Math.max(0, Math.min(duration, x / PIXELS_PER_SECOND));
      onSeek(t);
    },
    [onSeek, duration],
  );

  /* ── Double-click lane → add segment ────────────── */
  const handleLaneDoubleClick = useCallback(
    (trackId: string, e: React.MouseEvent<HTMLDivElement>) => {
      if (!onSegmentsChange) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const t = x / PIXELS_PER_SECOND;

      const trackSegs = segmentsByTrack.get(trackId) ?? [];
      const newStart = Math.max(0, t - 0.5);
      const newEnd = Math.min(duration, t + 0.5);

      let start = newStart;
      let end = newEnd;
      for (const seg of trackSegs) {
        if (start < seg.end && end > seg.start) {
          if (t < seg.start) {
            end = Math.min(end, seg.start);
          } else if (t > seg.end) {
            start = Math.max(start, seg.end);
          } else {
            return;
          }
        }
      }

      if (end - start < MIN_SEGMENT_DURATION) return;

      const newSeg: Segment = {
        id: `seg-${Date.now()}`,
        trackId,
        start,
        end,
        tagId: tags[0]?.id ?? 'tag-fire',
        name: '',
        prompt: '',
      };
      onSegmentsChange([...segments, newSeg]);
    },
    [onSegmentsChange, segments, segmentsByTrack, tags, duration],
  );

  /* ── Segment click → select ──────────────────────── */
  const handleSegmentClick = useCallback(
    (segId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      selectSegment(segId, e.shiftKey || e.metaKey || e.ctrlKey);
    },
    [selectSegment],
  );

  /* ── Segment right-click → delete ───────────────── */
  const handleSegmentDelete = useCallback(
    (segId: string, e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!onSegmentsChange) return;
      if (selection.segments.has(segId) && selection.segments.size > 1) {
        onSegmentsChange(segments.filter((s) => !selection.segments.has(s.id)));
        setSelection((prev) => ({ ...prev, segments: new Set() }));
      } else {
        onSegmentsChange(segments.filter((s) => s.id !== segId));
      }
    },
    [onSegmentsChange, segments, selection],
  );

  /* ── Segment resize ─────────────────────────────── */
  const handleResizeStart = useCallback(
    (segId: string, edge: 'start' | 'end', e: React.MouseEvent) => {
      e.stopPropagation();
      e.preventDefault();
      const seg = segments.find((s) => s.id === segId);
      if (!seg) return;
      setResizing({
        segId,
        edge,
        originX: e.clientX,
        originTime: edge === 'start' ? seg.start : seg.end,
      });
    },
    [segments],
  );

  useEffect(() => {
    if (!resizing) return;
    const handleMove = (e: MouseEvent) => {
      if (!onSegmentsChange) return;
      const dx = e.clientX - resizing.originX;
      const dt = dx / PIXELS_PER_SECOND;
      let newTime = Math.max(0, Math.min(duration, resizing.originTime + dt));

      onSegmentsChange(
        segments.map((s) => {
          if (s.id !== resizing.segId) return s;
          if (resizing.edge === 'start') {
            const start = Math.min(newTime, s.end - MIN_SEGMENT_DURATION);
            return { ...s, start: Math.max(0, start) };
          } else {
            const end = Math.max(newTime, s.start + MIN_SEGMENT_DURATION);
            return { ...s, end: Math.min(duration, end) };
          }
        }),
      );
    };
    const handleUp = () => setResizing(null);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [resizing, segments, onSegmentsChange, duration]);

  /* ── Keyframe double-click fade lane → add ──────── */
  const handleFadeLaneDoubleClick = useCallback(
    (trackId: string, e: React.MouseEvent<HTMLDivElement>) => {
      if (!onKeyframesChange) return;
      e.stopPropagation();
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const t = Math.max(0, Math.min(duration, x / PIXELS_PER_SECOND));
      const value = Math.max(0, Math.min(1, 1 - y / FADE_LANE_HEIGHT));
      const newKf: Keyframe = {
        id: `kf-${Date.now()}`,
        trackId,
        time: t,
        value: Math.round(value * 100) / 100,
      };
      onKeyframesChange([...keyframes, newKf]);
    },
    [onKeyframesChange, keyframes, duration],
  );

  /* ── Keyframe right-click → delete ──────────────── */
  const handleKeyframeDelete = useCallback(
    (kfId: string, e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!onKeyframesChange) return;
      if (selection.keyframes.has(kfId) && selection.keyframes.size > 1) {
        onKeyframesChange(keyframes.filter((k) => !selection.keyframes.has(k.id)));
        setSelection((prev) => ({ ...prev, keyframes: new Set() }));
      } else {
        onKeyframesChange(keyframes.filter((k) => k.id !== kfId));
      }
    },
    [onKeyframesChange, keyframes, selection],
  );

  /* ── Keyframe drag ── */
  const handleKeyframeDragStart = useCallback(
    (kfId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      e.preventDefault();
      selectKeyframe(kfId, e.shiftKey || e.metaKey || e.ctrlKey);
      setDraggingKf(kfId);
    },
    [selectKeyframe],
  );

  useEffect(() => {
    if (!draggingKf) return;
    const handleMove = (e: MouseEvent) => {
      if (!onKeyframesChange) return;
      const kf = keyframes.find((k) => k.id === draggingKf);
      if (!kf) return;
      const laneEl = document.querySelector(`[data-fade-lane="${kf.trackId}"]`) as HTMLElement | null;
      if (!laneEl) return;
      const rect = laneEl.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const t = Math.max(0, Math.min(duration, x / PIXELS_PER_SECOND));
      const value = Math.max(0, Math.min(1, 1 - y / FADE_LANE_HEIGHT));
      onKeyframesChange(
        keyframes.map((k) =>
          k.id === draggingKf
            ? { ...k, time: Math.round(t * 100) / 100, value: Math.round(value * 100) / 100 }
            : k,
        ),
      );
    };
    const handleUp = () => setDraggingKf(null);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [draggingKf, keyframes, onKeyframesChange, duration]);

  return (
    <div className="timeline-panel" ref={panelRef} tabIndex={-1}>
      <div className="timeline-header">
        <span>Timeline</span>
        {hasSelection && (
          <span className="selection-info">
            {selection.segments.size > 0 && `${selection.segments.size} seg`}
            {selection.segments.size > 0 && selection.keyframes.size > 0 && ', '}
            {selection.keyframes.size > 0 && `${selection.keyframes.size} kf`}
            {' selected'}
            <button className="selection-delete-btn" onClick={deleteSelection} title="Delete selected (Delete key)">Del</button>
          </span>
        )}
        <span className="timeline-hint">
          Click = select | Shift+click = multi | Delete = remove | Dbl-click lane = add
        </span>
      </div>

      {/* Ruler */}
      <div
        className="timeline-ruler"
        style={{ width: rulerWidth + LABEL_WIDTH + 8, minWidth: rulerWidth + LABEL_WIDTH + 8 }}
        onClick={handleRulerClick}
      >
        {rulerTicks.map(({ t, major }) => (
          <div
            key={t}
            className={`ruler-tick ${major ? 'major' : ''}`}
            style={{ left: LABEL_WIDTH + t * PIXELS_PER_SECOND }}
          />
        ))}
        {rulerTicks
          .filter(({ major }) => major)
          .map(({ t }) => (
            <span
              key={t}
              className="ruler-label"
              style={{ left: LABEL_WIDTH + t * PIXELS_PER_SECOND }}
            >
              {t}s
            </span>
          ))}
        <div className="playhead-ruler" style={{ left: playheadX }} />
      </div>

      {/* Tracks — one row per path */}
      <div className="timeline-tracks">
        {paths.map((path) => {
          const trackSegments = segmentsByTrack.get(path.id) ?? [];
          const trackKeyframes = keyframesByTrack.get(path.id) ?? [];
          const envelopePoints = buildFadeEnvelopePoints(
            trackKeyframes,
            duration,
            rulerWidth,
            FADE_LANE_HEIGHT,
          );

          const fadeMaskGradient = buildFadeMaskGradient(trackKeyframes, duration);
          const hasTexturedSegments = trackSegments.some(
            (seg) => tagById.get(seg.tagId)?.textureUrl
          );

          return (
            <div key={path.id} className="track-row" style={{ height: TRACK_HEIGHT }}>
              <div className="track-label">{path.name}</div>
              <div
                className="track-lane"
                style={{ width: rulerWidth, minWidth: rulerWidth, height: TRACK_HEIGHT }}
              >
                {/* ── Texture backdrop ── */}
                {hasTexturedSegments && (
                  <div
                    className="track-texture-backdrop"
                    style={{
                      WebkitMaskImage: fadeMaskGradient !== 'none' ? fadeMaskGradient : undefined,
                      maskImage: fadeMaskGradient !== 'none' ? fadeMaskGradient : undefined,
                    }}
                  >
                    {trackSegments.map((seg) => {
                      const tag = tagById.get(seg.tagId);
                      if (!tag?.textureUrl) return null;
                      const segLeft = seg.start * PIXELS_PER_SECOND;
                      const segWidth = (seg.end - seg.start) * PIXELS_PER_SECOND;
                      return (
                        <div
                          key={seg.id}
                          className="track-texture-strip"
                          style={{
                            left: segLeft,
                            width: Math.max(segWidth, 1),
                            backgroundImage: `url(${tag.textureUrl})`,
                            backgroundSize: `${TRACK_HEIGHT}px ${TRACK_HEIGHT}px`,
                          }}
                        />
                      );
                    })}
                  </div>
                )}

                {/* ── Segment blocks ── */}
                <div
                  className="segment-area"
                  onClick={clearSelection}
                  onDoubleClick={(e) => handleLaneDoubleClick(path.id, e)}
                >
                  {trackSegments.map((seg) => {
                    const tag = tagById.get(seg.tagId);
                    const textureUrl = tag?.textureUrl;
                    const left = seg.start * PIXELS_PER_SECOND;
                    const width = (seg.end - seg.start) * PIXELS_PER_SECOND;
                    const displayName = tag?.label || seg.tagId;
                    const isSelected = selection.segments.has(seg.id);
                    return (
                      <div
                        key={seg.id}
                        className={`segment ${isSelected ? 'selected' : ''}${textureUrl ? ' has-texture' : ''}`}
                        style={{
                          left,
                          width: Math.max(width, 24),
                          backgroundImage: textureUrl ? `url(${textureUrl})` : undefined,
                          backgroundSize: textureUrl ? `${SEGMENT_HEIGHT}px ${SEGMENT_HEIGHT}px` : undefined,
                          backgroundRepeat: textureUrl ? 'repeat' : undefined,
                          backgroundColor: textureUrl ? undefined : (tag?.color ?? DEFAULT_TAG_COLOR) + 'cc',
                          color: '#fff',
                        }}
                        title={`${displayName} (${seg.start.toFixed(1)}–${seg.end.toFixed(1)}s)`}
                        onClick={(e) => handleSegmentClick(seg.id, e)}
                        onContextMenu={(e) => handleSegmentDelete(seg.id, e)}
                      >
                        <div
                          className="segment-resize-handle left"
                          onMouseDown={(e) => handleResizeStart(seg.id, 'start', e)}
                        />
                        <span className="segment-label">{displayName}</span>
                        <div
                          className="segment-resize-handle right"
                          onMouseDown={(e) => handleResizeStart(seg.id, 'end', e)}
                        />
                      </div>
                    );
                  })}
                </div>

                {/* ── Fade envelope lane ── */}
                <div
                  className="fade-lane"
                  data-fade-lane={path.id}
                  onClick={clearSelection}
                  onDoubleClick={(e) => handleFadeLaneDoubleClick(path.id, e)}
                >
                  {trackKeyframes.length > 0 && (
                    <svg
                      className="fade-envelope-svg"
                      width={rulerWidth}
                      height={FADE_LANE_HEIGHT}
                    >
                      <polyline
                        points={`0,${FADE_LANE_HEIGHT} ${envelopePoints} ${rulerWidth},${FADE_LANE_HEIGHT}`}
                        fill="rgba(245, 158, 11, 0.12)"
                        stroke="none"
                      />
                      <polyline
                        points={envelopePoints}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="1.5"
                        strokeLinejoin="round"
                      />
                    </svg>
                  )}

                  {trackKeyframes.map((kf) => {
                    const kfX = kf.time * PIXELS_PER_SECOND;
                    const kfY = FADE_LANE_HEIGHT - kf.value * (FADE_LANE_HEIGHT - 2);
                    const kfSelected = selection.keyframes.has(kf.id);
                    return (
                      <div
                        key={kf.id}
                        className={`keyframe-dot${draggingKf === kf.id ? ' dragging' : ''}${kfSelected ? ' selected' : ''}`}
                        style={{ left: kfX, top: kfY }}
                        title={`t=${kf.time.toFixed(2)}s  value=${kf.value.toFixed(2)}`}
                        onContextMenu={(e) => handleKeyframeDelete(kf.id, e)}
                        onMouseDown={(e) => handleKeyframeDragStart(kf.id, e)}
                      />
                    );
                  })}

                  <span className="fade-lane-label">fade</span>
                </div>

                <div className="playhead-line" style={{ left: currentTime * PIXELS_PER_SECOND }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
