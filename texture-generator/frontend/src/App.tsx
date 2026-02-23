import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import PathView from './components/PathView';
import type { PathViewHandle } from './components/PathView';
import Timeline from './components/Timeline';
import PlaybackControls from './components/PlaybackControls';
import VideoPanel from './components/VideoPanel';
import TagPanel from './components/TagPanel';
import MenuBar from './components/MenuBar';
import {
  mockPaths,
  mockSegments,
  mockTags,
  mockKeyframes,
} from './data/mockData';
import { saveState, loadState, exportStateToFile, importStateFromFile } from './utils/persistence';
import { useUndoRedo } from './hooks/useUndoRedo';
import type { Tag, Segment } from './types';

const DURATION = 8;

const TAG_PALETTE = [
  '#ef4444', '#3b82f6', '#22c55e', '#a1a1aa',
  '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6',
  '#f97316', '#6366f1',
];

/** Ensure all tags referenced by segments exist in the tags array, and all tags have a color. */
function syncTags(tags: Tag[], segments: Segment[]): Tag[] {
  let result = tags.map((t, i) =>
    t.color ? t : { ...t, color: TAG_PALETTE[i % TAG_PALETTE.length] }
  );

  const existingIds = new Set(result.map((t) => t.id));
  const usedColors = new Set(result.map((t) => t.color));

  for (const seg of segments) {
    if (!existingIds.has(seg.tagId)) {
      existingIds.add(seg.tagId);
      const color = TAG_PALETTE.find((c) => !usedColors.has(c))
        ?? TAG_PALETTE[result.length % TAG_PALETTE.length];
      usedColors.add(color);
      const label = seg.tagId
        .replace(/^tag-/, '')
        .replace(/-\d+$/, '')
        .replace(/-/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase());
      result.push({
        id: seg.tagId,
        label,
        color,
        textureId: null,
        textureUrl: null,
        displacementUrl: null,
        normalUrl: null,
        referenceImageUrl: null,
        ipAdapterScale: 0.5,
      });
    }
  }

  return result;
}

function getInitialStateSync() {
  // Load structure from localStorage synchronously (no images yet)
  try {
    const raw = localStorage.getItem('texture-generator-state');
    if (raw) {
      const saved = JSON.parse(raw) as import('./types').AppState;
      const synced = syncTags(saved.tags, saved.segments);
      return { ...saved, tags: synced };
    }
  } catch (_) { /* fall through */ }
  return {
    paths: mockPaths,
    tags: syncTags(mockTags, mockSegments),
    segments: mockSegments,
    keyframes: mockKeyframes,
  };
}

/* ── Drag-resize hook ──────────────────────────────────────────────── */
function useResize(
  initial: number,
  direction: 'horizontal' | 'vertical',
  min: number,
  max: number,
) {
  const [size, setSize] = useState(initial);
  const dragging = useRef(false);
  const startPos = useRef(0);
  const startSize = useRef(0);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;
      startPos.current = direction === 'horizontal' ? e.clientY : e.clientX;
      startSize.current = size;
      document.body.style.cursor = direction === 'horizontal' ? 'row-resize' : 'col-resize';
      document.body.style.userSelect = 'none';
    },
    [size, direction],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const pos = direction === 'horizontal' ? e.clientY : e.clientX;
      const delta = pos - startPos.current;
      setSize(Math.min(max, Math.max(min, startSize.current + delta)));
    };
    const onUp = () => {
      if (!dragging.current) return;
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [direction, min, max]);

  return { size, onMouseDown };
}

export default function App() {
  const initial = useRef(getInitialStateSync());
  const [tags, setTags] = useState<Tag[]>(initial.current.tags);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const rafRef = useRef<number>(0);
  const lastFrameRef = useRef<number>(0);
  const pathViewRef = useRef<PathViewHandle>(null);
  const [maxDisplacement, setMaxDisplacement] = useState(1.0);
  const [showTextures, setShowTextures] = useState(true);
  const [textureDensity, setTextureDensity] = useState(1.0);
  const [dispCompression, setDispCompression] = useState(1.0);

  // Hydrate image data from IndexedDB after mount
  useEffect(() => {
    loadState().then((saved) => {
      if (saved) {
        setTags((prev) =>
          prev.map((t) => {
            const restored = saved.tags.find((s) => s.id === t.id);
            return restored
              ? { ...t, textureUrl: restored.textureUrl, displacementUrl: restored.displacementUrl, normalUrl: restored.normalUrl, referenceImageUrl: restored.referenceImageUrl }
              : t;
          })
        );
      }
    });
  }, []);

  /* ── Resize handles ── */
  const topResize = useResize(Math.round(window.innerHeight * 0.55), 'horizontal', 160, window.innerHeight - 200);
  const videoResize = useResize(280, 'vertical', 180, 600);
  const tagResize = useResize(260, 'vertical', 180, 500);

  const {
    segments,
    keyframes,
    setSegments,
    setKeyframes,
    replaceAll,
    undo,
    redo,
    canUndo,
    canRedo,
  } = useUndoRedo({
    segments: initial.current.segments,
    keyframes: initial.current.keyframes,
  });

  const syncedTags = useMemo(() => {
    return syncTags(tags, segments);
  }, [tags, segments]);

  useEffect(() => {
    if (syncedTags.length !== tags.length || !syncedTags.every((t, i) => t === tags[i])) {
      setTags(syncedTags);
    }
  }, [syncedTags, tags]);

  useEffect(() => {
    const timer = setTimeout(() => {
      saveState({ paths: mockPaths, tags: syncedTags, segments, keyframes });
    }, 500);
    return () => clearTimeout(timer);
  }, [syncedTags, segments, keyframes]);

  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod || e.key.toLowerCase() !== 'z') return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [undo, redo]);

  const play = useCallback(() => {
    setIsPlaying(true);
    lastFrameRef.current = performance.now();
  }, []);

  const pause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const togglePlay = useCallback(() => {
    if (isPlaying) pause();
    else play();
  }, [isPlaying, play, pause]);

  const seek = useCallback((t: number) => {
    setCurrentTime(Math.max(0, Math.min(DURATION, t)));
  }, []);

  const handleExport = useCallback(() => {
    exportStateToFile({ paths: mockPaths, tags: syncedTags, segments, keyframes });
  }, [syncedTags, segments, keyframes]);

  const handleImport = useCallback(async () => {
    try {
      const state = await importStateFromFile();
      setTags(syncTags(state.tags, state.segments));
      replaceAll({ segments: state.segments, keyframes: state.keyframes });
    } catch (e) {
      console.error('Import failed:', e);
    }
  }, [replaceAll]);

  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }
    const tick = (now: number) => {
      const dt = (now - lastFrameRef.current) / 1000;
      lastFrameRef.current = now;
      setCurrentTime((prev) => {
        const next = prev + dt;
        if (next >= DURATION) {
          setIsPlaying(false);
          return DURATION;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isPlaying]);

  return (
    <div className="app">
      {/* ── Menu bar ── */}
      <MenuBar
        onExportProject={handleExport}
        onImportProject={handleImport}
        onExportMesh={() => pathViewRef.current?.exportMesh()}
        onExportUSDZ={() => pathViewRef.current?.exportUSDZ()}
        onExportOBJ={() => pathViewRef.current?.exportOBJ()}
      />

      {/* ── Top region: Video + 3D ── */}
      <div className="top-region" style={{ height: topResize.size }}>
        <div className="video-wrapper" style={{ width: videoResize.size }}>
          <VideoPanel currentTime={currentTime} isPlaying={isPlaying} />
        </div>
        <div className="resize-handle-v" onMouseDown={videoResize.onMouseDown} />
        <div className="viewport-wrapper">
          <PathView
            ref={pathViewRef}
            paths={mockPaths}
            segments={segments}
            keyframes={keyframes}
            tags={syncedTags}
            currentTime={currentTime}
            duration={DURATION}
            maxDisplacement={maxDisplacement}
            showTextures={showTextures}
            textureDensity={textureDensity}
            dispCompression={dispCompression}
          />
        </div>
      </div>

      {/* ── Horizontal drag handle ── */}
      <div className="resize-handle-h" onMouseDown={topResize.onMouseDown} />

      {/* ── Playback controls ── */}
      <PlaybackControls
        isPlaying={isPlaying}
        currentTime={currentTime}
        duration={DURATION}
        onTogglePlay={togglePlay}
        onSeek={seek}
        canUndo={canUndo}
        canRedo={canRedo}
        onUndo={undo}
        onRedo={redo}
      />

      {/* ── Bottom region: Tags + Timeline ── */}
      <div className="bottom-region">
        <div className="tag-wrapper" style={{ width: tagResize.size }}>
          <TagPanel tags={syncedTags} onTagsChange={setTags} />
          <div className="sidebar-actions">
            <button
              className={`action-btn texture-toggle-btn${showTextures ? ' active' : ''}`}
              onClick={() => setShowTextures((v) => !v)}
              title={showTextures ? 'Hide displacement textures' : 'Show displacement textures'}
            >
              {showTextures ? 'Tex ON' : 'Tex OFF'}
            </button>
            <label className="displacement-slider-label">Disp</label>
            <input
              type="range"
              className="displacement-slider"
              min={0}
              max={2}
              step={0.01}
              value={maxDisplacement}
              onChange={(e) => setMaxDisplacement(parseFloat(e.target.value))}
            />
            <span className="displacement-slider-value">{maxDisplacement.toFixed(2)}</span>
            <label className="displacement-slider-label">Density</label>
            <input
              type="range"
              className="displacement-slider"
              min={0.1}
              max={3}
              step={0.05}
              value={textureDensity}
              onChange={(e) => setTextureDensity(parseFloat(e.target.value))}
            />
            <span className="displacement-slider-value">{textureDensity.toFixed(2)}</span>
            <label className="displacement-slider-label">Compress</label>
            <input
              type="range"
              className="displacement-slider"
              min={0.1}
              max={2}
              step={0.05}
              value={dispCompression}
              onChange={(e) => setDispCompression(parseFloat(e.target.value))}
            />
            <span className="displacement-slider-value">{dispCompression.toFixed(2)}</span>
          </div>
        </div>
        <div className="resize-handle-v" onMouseDown={tagResize.onMouseDown} />
        <Timeline
          paths={mockPaths}
          segments={segments}
          tags={syncedTags}
          keyframes={keyframes}
          duration={DURATION}
          currentTime={currentTime}
          onSeek={seek}
          onSegmentsChange={setSegments}
          onKeyframesChange={setKeyframes}
        />
      </div>
    </div>
  );
}
