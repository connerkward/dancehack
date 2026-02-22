import { useState, useCallback, useEffect, useRef, lazy, Suspense } from 'react';
import type { Tag } from '../types';

const DisplacementPreview = lazy(() => import('./DisplacementPreview'));

const GENERATOR_URL = 'http://localhost:8100';

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

const TAG_PALETTE = [
  '#ef4444', '#3b82f6', '#22c55e', '#a1a1aa',
  '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6',
  '#f97316', '#6366f1',
];

type GenStatus = { phase: 'texture' | 'displacement' | 'done' | 'error'; message: string } | null;

export default function TagPanel({
  tags,
  onTagsChange,
}: {
  tags: Tag[];
  onTagsChange: (update: Tag[] | ((prev: Tag[]) => Tag[])) => void;
}) {
  const [generating, setGenerating] = useState<string | null>(null);
  const [status, setStatus] = useState<Record<string, GenStatus>>({});
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [preview, setPreview] = useState<{ tagId: string; url: string; label: string; displacementUrl?: string | null; normalUrl?: string | null } | null>(null);
  const [dispScale, setDispScale] = useState(0.5);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editLabel, setEditLabel] = useState('');
  const [editPrompt, setEditPrompt] = useState('');
  const [addingNew, setAddingNew] = useState(false);
  const [newLabel, setNewLabel] = useState('');
  const [newPrompt, setNewPrompt] = useState('');
  const editRef = useRef<HTMLDivElement>(null);
  const newRef = useRef<HTMLDivElement>(null);

  // Close preview on Escape
  useEffect(() => {
    if (!preview) return;
    const handle = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setPreview(null);
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [preview]);

  // Click outside to close edit / add-new
  useEffect(() => {
    if (!editingId && !addingNew) return;
    const handle = (e: MouseEvent) => {
      if (editingId && editRef.current && !editRef.current.contains(e.target as Node)) {
        setEditingId(null);
      }
      if (addingNew && newRef.current && !newRef.current.contains(e.target as Node)) {
        setAddingNew(false);
        setNewLabel('');
        setNewPrompt('');
      }
    };
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, [editingId, addingNew]);

  const setTagStatus = (tagId: string, s: GenStatus) =>
    setStatus((prev) => ({ ...prev, [tagId]: s }));

  const updateTag = useCallback(
    (tagId: string, updates: Partial<Tag>) => {
      onTagsChange((prev) =>
        prev.map((t) => (t.id === tagId ? { ...t, ...updates } : t))
      );
    },
    [onTagsChange]
  );

  const deleteTag = useCallback(
    (tagId: string) => {
      onTagsChange((prev) => prev.filter((t) => t.id !== tagId));
    },
    [onTagsChange]
  );

  const startEditing = useCallback((tag: Tag) => {
    setEditingId(tag.id);
    setEditLabel(tag.label);
    setEditPrompt(tag.prompt ?? '');
  }, []);

  const saveEdit = useCallback(() => {
    if (!editingId) return;
    const label = editLabel.trim();
    if (!label) return;
    updateTag(editingId, { label, prompt: editPrompt.trim() });
    setEditingId(null);
  }, [editingId, editLabel, editPrompt, updateTag]);

  const addNewTag = useCallback(() => {
    const label = newLabel.trim();
    if (!label) return;
    const usedColors = new Set(tags.map((t) => t.color));
    const color = TAG_PALETTE.find((c) => !usedColors.has(c)) ?? TAG_PALETTE[tags.length % TAG_PALETTE.length];
    const newTag: Tag = {
      id: `tag-${label.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
      label,
      prompt: newPrompt.trim(),
      color,
      textureId: null,
      textureUrl: null,
      displacementUrl: null,
      normalUrl: null,
    };
    onTagsChange((prev) => [...prev, newTag]);
    setAddingNew(false);
    setNewLabel('');
    setNewPrompt('');
  }, [newLabel, newPrompt, tags, onTagsChange]);

  const generateTexture = useCallback(
    async (tag: Tag) => {
      setGenerating(tag.id);
      setProgress((prev) => ({ ...prev, [tag.id]: 0 }));
      setTagStatus(tag.id, { phase: 'texture', message: 'Generating texture...' });
      try {
        const prompt = tag.prompt?.trim() || tag.label;
        const form = new FormData();
        form.append('prompt', prompt);
        form.append('width', '1024');
        form.append('height', '1024');
        form.append('steps', '30');
        form.append('guidance', '7');

        // Stream progress via SSE
        const res = await fetch(`${GENERATOR_URL}/generate-stream`, {
          method: 'POST',
          body: form,
        });

        if (!res.ok) throw new Error(`Generator ${res.status}`);

        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let b64Image: string | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split('\n\n');
          buffer = blocks.pop()!;

          for (const block of blocks) {
            const eventMatch = block.match(/event: (\w+)/);
            const dataMatch = block.match(/data: (.+)/s);
            if (!eventMatch || !dataMatch) continue;

            const event = eventMatch[1];
            const data = JSON.parse(dataMatch[1]);

            if (event === 'progress') {
              const pct = data.step / data.total;
              setProgress((prev) => ({ ...prev, [tag.id]: pct }));
              setTagStatus(tag.id, {
                phase: 'texture',
                message: `Step ${data.step}/${data.total}`,
              });
            } else if (event === 'complete') {
              b64Image = data.image;
            } else if (event === 'error') {
              throw new Error(data.message);
            }
          }
        }

        if (!b64Image) throw new Error('No image received');

        // Store as data URL so it persists in localStorage across reloads
        const textureUrl = `data:image/png;base64,${b64Image}`;

        // Convert to blob for sending to displacement/normal endpoints
        const byteString = atob(b64Image);
        const bytes = new Uint8Array(byteString.length);
        for (let i = 0; i < byteString.length; i++) bytes[i] = byteString.charCodeAt(i);
        const blob = new Blob([bytes], { type: 'image/png' });

        // Displacement
        setProgress((prev) => ({ ...prev, [tag.id]: 1 }));
        setTagStatus(tag.id, { phase: 'displacement', message: 'Generating displacement...' });
        let displacementUrl: string | null = null;
        let normalUrl: string | null = null;
        try {
          const dispForm = new FormData();
          dispForm.append('image', blob, 'texture.png');
          const dispRes = await fetch(`${GENERATOR_URL}/displacement`, {
            method: 'POST',
            body: dispForm,
          });
          if (dispRes.ok) {
            const dispBlob = await dispRes.blob();
            displacementUrl = await blobToDataUrl(dispBlob);
          }
        } catch (err) {
          console.warn('Displacement generation failed:', err);
        }

        // Normal map
        try {
          const normForm = new FormData();
          normForm.append('image', blob, 'texture.png');
          const normRes = await fetch(`${GENERATOR_URL}/normal`, {
            method: 'POST',
            body: normForm,
          });
          if (normRes.ok) {
            const normBlob = await normRes.blob();
            normalUrl = await blobToDataUrl(normBlob);
          }
        } catch (err) {
          console.warn('Normal map generation failed:', err);
        }

        onTagsChange((prev) =>
          prev.map((t) =>
            t.id === tag.id ? { ...t, textureUrl, displacementUrl, normalUrl } : t
          )
        );
        setTagStatus(tag.id, { phase: 'done', message: 'Done' });
        setTimeout(() => setTagStatus(tag.id, null), 2000);
      } catch (err) {
        console.error('Texture generation failed:', err);
        setTagStatus(tag.id, { phase: 'error', message: 'Failed' });
        setTimeout(() => setTagStatus(tag.id, null), 3000);
      } finally {
        setGenerating(null);
        setProgress((prev) => {
          const next = { ...prev };
          delete next[tag.id];
          return next;
        });
      }
    },
    [onTagsChange]
  );

  return (
    <div className="tag-panel">
      <div className="tag-panel-header">
        <span>Tags</span>
        <button
          className="tag-add-btn"
          onClick={() => { setAddingNew(true); setEditingId(null); }}
          title="Add new tag"
        >+</button>
      </div>

      {/* Add new tag form */}
      {addingNew && (
        <div className="tag-edit-form" ref={newRef}>
          <input
            autoFocus
            className="tag-edit-input"
            placeholder="Tag name..."
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') addNewTag(); if (e.key === 'Escape') { setAddingNew(false); setNewLabel(''); setNewPrompt(''); } }}
          />
          <textarea
            className="tag-edit-textarea"
            placeholder="Texture prompt..."
            rows={2}
            value={newPrompt}
            onChange={(e) => setNewPrompt(e.target.value)}
          />
          <div className="tag-edit-actions">
            <button className="tag-edit-save" onClick={addNewTag}>Add</button>
            <button className="tag-edit-cancel" onClick={() => { setAddingNew(false); setNewLabel(''); setNewPrompt(''); }}>Cancel</button>
          </div>
        </div>
      )}

      {tags.map((tag) => {
        const s = status[tag.id];
        const isActive = generating === tag.id;
        const isEditing = editingId === tag.id;

        return (
          <div key={tag.id} className="tag-item">
            <div
              className={`tag-color-swatch${isActive ? ' generating' : ''}${tag.textureUrl ? ' clickable' : ''}`}
              style={{
                backgroundImage: tag.textureUrl ? `url(${tag.textureUrl})` : undefined,
                backgroundColor: tag.textureUrl ? undefined : tag.color,
                '--progress': progress[tag.id] ?? 0,
              } as React.CSSProperties}
              onClick={() => { if (tag.textureUrl) { setPreview({ tagId: tag.id, url: tag.textureUrl, label: tag.label, displacementUrl: tag.displacementUrl, normalUrl: tag.normalUrl }); setDispScale(tag.displacementScale ?? 0.5); } }}
            />
            {tag.displacementUrl && (
              <div
                className="tag-displacement-swatch"
                style={{ backgroundImage: `url(${tag.displacementUrl})` }}
                title="Displacement map"
              />
            )}

            {isEditing ? (
              <div className="tag-edit-form tag-edit-inline" ref={editRef}>
                <input
                  autoFocus
                  className="tag-edit-input"
                  placeholder="Tag name..."
                  value={editLabel}
                  onChange={(e) => setEditLabel(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') saveEdit(); if (e.key === 'Escape') setEditingId(null); }}
                />
                <textarea
                  className="tag-edit-textarea"
                  placeholder="Texture prompt..."
                  rows={2}
                  value={editPrompt}
                  onChange={(e) => setEditPrompt(e.target.value)}
                />
                <div className="tag-edit-actions">
                  <button className="tag-edit-save" onClick={saveEdit}>Save</button>
                  <button className="tag-edit-cancel" onClick={() => setEditingId(null)}>Cancel</button>
                  <button className="tag-edit-delete" onClick={() => { deleteTag(tag.id); setEditingId(null); }}>Del</button>
                </div>
              </div>
            ) : (
              <div className="tag-label-group" onClick={() => startEditing(tag)}>
                <span className="tag-label">{tag.label}</span>
                {tag.prompt && <span className="tag-prompt-hint">{tag.prompt}</span>}
                {s && (
                  <span className={`tag-status tag-status-${s.phase}`}>
                    {s.message}
                  </span>
                )}
              </div>
            )}

            {!isEditing && (
              <button
                className="tag-generate-btn"
                disabled={isActive}
                onClick={() => generateTexture(tag)}
              >
                {isActive ? '...' : 'Gen'}
              </button>
            )}
          </div>
        );
      })}

      {/* Texture preview floating panel */}
      {preview && (
        <div className="texture-preview-panel">
          <div className="texture-preview-panel-header">
            <span className="texture-preview-label">{preview.label}</span>
            <button className="texture-preview-close" onClick={() => setPreview(null)}>×</button>
          </div>
          <div className="texture-preview-images">
            <div className="texture-preview-main">
              <img src={preview.url} alt={preview.label} />
              <span className="texture-preview-badge">Color</span>
            </div>
            {preview.displacementUrl && (
              <div className="texture-preview-main">
                <div className="texture-preview-3d">
                  <Suspense fallback={<div className="texture-preview-loading">Loading 3D...</div>}>
                    <DisplacementPreview
                      textureUrl={preview.url}
                      displacementUrl={preview.displacementUrl}
                      normalUrl={preview.normalUrl}
                      displacementScale={dispScale}
                    />
                  </Suspense>
                </div>
                <span className="texture-preview-badge">3D</span>
              </div>
            )}
          </div>
          {preview.displacementUrl && (
            <div className="texture-preview-slider-row">
              <label className="texture-preview-slider-label">Depth</label>
              <input
                className="texture-preview-slider"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={dispScale}
                onChange={(e) => { const v = parseFloat(e.target.value); setDispScale(v); if (preview) updateTag(preview.tagId, { displacementScale: v }); }}
              />
              <span className="texture-preview-slider-value">{dispScale.toFixed(2)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
