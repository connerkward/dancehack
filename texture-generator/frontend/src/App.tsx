import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import PathView from './components/PathView';
import Timeline from './components/Timeline';
import PlaybackControls from './components/PlaybackControls';
import VideoPanel from './components/VideoPanel';
import TagPanel from './components/TagPanel';
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
  // Migrate: ensure every tag has a color
  let result = tags.map((t, i) =>
    t.color ? t : { ...t, color: TAG_PALETTE[i % TAG_PALETTE.length] }
  );

  // Find tags referenced by segments but missing from the list
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
      });
    }
  }

  return result;
}

function getInitialState() {
  const saved = loadState();
  const base = saved ?? {
    paths: mockPaths,
    tags: mockTags,
    segments: mockSegments,
    keyframes: mockKeyframes,
  };
  // Always sync tags with segments on load
  return { ...base, tags: syncTags(base.tags, base.segments) };
}

export default function App() {
  const initial = useRef(getInitialState());
  const [tags, setTags] = useState<Tag[]>(initial.current.tags);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const rafRef = useRef<number>(0);
  const lastFrameRef = useRef<number>(0);

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

  // Always derive the correct tag set from raw tags + segments
  const syncedTags = useMemo(() => syncTags(tags, segments), [tags, segments]);

  // Keep raw state in sync so functional updaters see the full tag list
  useEffect(() => {
    if (syncedTags.length !== tags.length || !syncedTags.every((t, i) => t === tags[i])) {
      setTags(syncedTags);
    }
  }, [syncedTags, tags]);

  // Auto-save on change (persist synced tags so localStorage is never missing tags)
  useEffect(() => {
    const timer = setTimeout(() => {
      saveState({ paths: mockPaths, tags: syncedTags, segments, keyframes });
    }, 500);
    return () => clearTimeout(timer);
  }, [syncedTags, segments, keyframes]);

  // Undo / Redo keyboard shortcuts
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
      <div className="top-panel">
        <div className="left-sidebar">
          <VideoPanel currentTime={currentTime} isPlaying={isPlaying} />
          <TagPanel tags={syncedTags} onTagsChange={setTags} />
          <div className="sidebar-actions">
            <button className="action-btn" onClick={handleExport}>Export</button>
            <button className="action-btn" onClick={handleImport}>Import</button>
          </div>
        </div>
        <PathView
          paths={mockPaths}
          segments={segments}
          keyframes={keyframes}
          tags={syncedTags}
          currentTime={currentTime}
          duration={DURATION}
        />
      </div>
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
        onTagsChange={setTags}
      />
    </div>
  );
}
