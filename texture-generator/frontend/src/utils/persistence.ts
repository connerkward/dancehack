import type { AppState, Tag } from '../types';

const STORAGE_KEY = 'texture-generator-state';
const DB_NAME = 'texture-generator';
const DB_VERSION = 1;
const IMAGE_STORE = 'images';

/* ── IndexedDB for image data ─────────────────────────────────────── */

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IMAGE_STORE)) {
        db.createObjectStore(IMAGE_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbPut(db: IDBDatabase, key: string, value: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IMAGE_STORE, 'readwrite');
    tx.objectStore(IMAGE_STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function idbGet(db: IDBDatabase, key: string): Promise<string | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IMAGE_STORE, 'readonly');
    const req = tx.objectStore(IMAGE_STORE).get(key);
    req.onsuccess = () => resolve(req.result as string | undefined);
    req.onerror = () => reject(req.error);
  });
}

function idbDelete(db: IDBDatabase, key: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IMAGE_STORE, 'readwrite');
    tx.objectStore(IMAGE_STORE).delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

const IMAGE_FIELDS = ['textureUrl', 'displacementUrl', 'normalUrl', 'referenceImageUrl'] as const;

/** Save all image data URLs from tags into IndexedDB. */
async function saveImages(tags: Tag[]): Promise<void> {
  const db = await openDB();
  for (const tag of tags) {
    for (const field of IMAGE_FIELDS) {
      const key = `${tag.id}:${field}`;
      const val = tag[field];
      if (val) {
        await idbPut(db, key, val);
      } else {
        await idbDelete(db, key);
      }
    }
  }
  db.close();
}

/** Restore image data URLs from IndexedDB into tags. */
async function loadImages(tags: Tag[]): Promise<Tag[]> {
  const db = await openDB();
  const result: Tag[] = [];
  for (const tag of tags) {
    const updates: Partial<Tag> = {};
    for (const field of IMAGE_FIELDS) {
      const key = `${tag.id}:${field}`;
      const val = await idbGet(db, key);
      if (val) {
        (updates as Record<string, string>)[field] = val;
      }
    }
    result.push({ ...tag, ...updates });
  }
  db.close();
  return result;
}

/** Strip image data from tags before saving to localStorage. */
function stripImages(tags: Tag[]): Tag[] {
  return tags.map((t) => ({
    ...t,
    textureUrl: t.textureUrl ? '__idb__' : null,
    displacementUrl: t.displacementUrl ? '__idb__' : null,
    normalUrl: t.normalUrl ? '__idb__' : null,
    referenceImageUrl: t.referenceImageUrl ? '__idb__' : null,
  }));
}

/* ── Public API ────────────────────────────────────────────────────── */

export function saveState(state: AppState): void {
  try {
    const lite = { ...state, tags: stripImages(state.tags) };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(lite));
  } catch (e) {
    console.error('Failed to save state:', e);
  }
  // Save images to IndexedDB (fire-and-forget)
  saveImages(state.tags).catch((e) => console.error('Failed to save images:', e));
}

export async function loadState(): Promise<AppState | null> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const state = JSON.parse(raw) as AppState;
    state.tags = await loadImages(state.tags);
    return state;
  } catch (e) {
    console.error('Failed to load state:', e);
    return null;
  }
}

export function exportStateToFile(state: AppState): void {
  const json = JSON.stringify(state, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'texture-generator-project.json';
  a.click();
  URL.revokeObjectURL(url);
}

export function importStateFromFile(): Promise<AppState> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) return reject(new Error('No file selected'));
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const state = JSON.parse(reader.result as string) as AppState;
          resolve(state);
        } catch (e) {
          reject(e);
        }
      };
      reader.readAsText(file);
    };
    input.click();
  });
}
