import { useState, useCallback, useRef } from 'react';
import type { Segment, Keyframe } from '../types';

export type Snapshot = {
  segments: Segment[];
  keyframes: Keyframe[];
};

const MAX_HISTORY = 80;
const BATCH_WINDOW_MS = 60;

export function useUndoRedo(initial: Snapshot) {
  const [history, setHistory] = useState<{
    past: Snapshot[];
    present: Snapshot;
    future: Snapshot[];
  }>({
    past: [],
    present: initial,
    future: [],
  });

  const lastPushTime = useRef(0);

  const setSegments = useCallback((segments: Segment[]) => {
    const now = performance.now();
    const shouldBatch = now - lastPushTime.current < BATCH_WINDOW_MS;
    lastPushTime.current = now;
    setHistory((h) => {
      if (shouldBatch) {
        return { ...h, present: { ...h.present, segments }, future: [] };
      }
      return {
        past: [...h.past.slice(-(MAX_HISTORY - 1)), h.present],
        present: { ...h.present, segments },
        future: [],
      };
    });
  }, []);

  const setKeyframes = useCallback((keyframes: Keyframe[]) => {
    const now = performance.now();
    const shouldBatch = now - lastPushTime.current < BATCH_WINDOW_MS;
    lastPushTime.current = now;
    setHistory((h) => {
      if (shouldBatch) {
        return { ...h, present: { ...h.present, keyframes }, future: [] };
      }
      return {
        past: [...h.past.slice(-(MAX_HISTORY - 1)), h.present],
        present: { ...h.present, keyframes },
        future: [],
      };
    });
  }, []);

  /** Replace everything (e.g. after import). Pushes current to history. */
  const replaceAll = useCallback((snapshot: Snapshot) => {
    lastPushTime.current = 0;
    setHistory((h) => ({
      past: [...h.past.slice(-(MAX_HISTORY - 1)), h.present],
      present: snapshot,
      future: [],
    }));
  }, []);

  const undo = useCallback(() => {
    setHistory((h) => {
      if (h.past.length === 0) return h;
      const prev = h.past[h.past.length - 1];
      return {
        past: h.past.slice(0, -1),
        present: prev,
        future: [h.present, ...h.future],
      };
    });
  }, []);

  const redo = useCallback(() => {
    setHistory((h) => {
      if (h.future.length === 0) return h;
      const next = h.future[0];
      return {
        past: [...h.past, h.present],
        present: next,
        future: h.future.slice(1),
      };
    });
  }, []);

  return {
    segments: history.present.segments,
    keyframes: history.present.keyframes,
    setSegments,
    setKeyframes,
    replaceAll,
    undo,
    redo,
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
  };
}
