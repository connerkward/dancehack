import { useState, useCallback, useEffect, lazy, Suspense } from 'react';
import type { Tag } from '../types';

const DisplacementPreview = lazy(() => import('./DisplacementPreview'));

const GENERATOR_URL = 'http://localhost:8100';

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
  const [preview, setPreview] = useState<{ url: string; label: string; displacementUrl?: string | null } | null>(null);
  const [dispScale, setDispScale] = useState(0.3);

  // Close preview on Escape
  useEffect(() => {
    if (!preview) return;
    const handle = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setPreview(null);
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [preview]);

  const setTagStatus = (tagId: string, s: GenStatus) =>
    setStatus((prev) => ({ ...prev, [tagId]: s }));

  const generateTexture = useCallback(
    async (tag: Tag) => {
      setGenerating(tag.id);
      setTagStatus(tag.id, { phase: 'texture', message: 'Generating texture...' });
      try {
        const form = new FormData();
        form.append('prompt', tag.label);
        form.append('width', '1024');
        form.append('height', '1024');
        form.append('steps', '30');
        form.append('guidance', '7');

        const res = await fetch(`${GENERATOR_URL}/generate`, {
          method: 'POST',
          body: form,
        });

        if (!res.ok) throw new Error(`Generator ${res.status}`);

        const blob = await res.blob();
        const textureUrl = URL.createObjectURL(blob);

        // Displacement
        setTagStatus(tag.id, { phase: 'displacement', message: 'Generating displacement...' });
        let displacementUrl: string | null = null;
        try {
          const dispForm = new FormData();
          dispForm.append('image', blob, 'texture.png');
          const dispRes = await fetch(`${GENERATOR_URL}/displacement`, {
            method: 'POST',
            body: dispForm,
          });
          if (dispRes.ok) {
            const dispBlob = await dispRes.blob();
            displacementUrl = URL.createObjectURL(dispBlob);
          }
        } catch (err) {
          console.warn('Displacement generation failed:', err);
        }

        onTagsChange((prev) =>
          prev.map((t) =>
            t.id === tag.id ? { ...t, textureUrl, displacementUrl } : t
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
      }
    },
    [onTagsChange]
  );

  return (
    <div className="tag-panel">
      <div className="tag-panel-header">Tags</div>
      {tags.map((tag) => {
        const s = status[tag.id];
        const isActive = generating === tag.id;
        return (
          <div key={tag.id} className="tag-item">
            <div
              className={`tag-color-swatch${isActive ? ' generating' : ''}${tag.textureUrl ? ' clickable' : ''}`}
              style={{
                backgroundImage: tag.textureUrl ? `url(${tag.textureUrl})` : undefined,
                backgroundColor: tag.textureUrl ? undefined : tag.color,
              }}
              onClick={() => tag.textureUrl && setPreview({ url: tag.textureUrl, label: tag.label, displacementUrl: tag.displacementUrl })}
            />
            {tag.displacementUrl && (
              <div
                className="tag-displacement-swatch"
                style={{ backgroundImage: `url(${tag.displacementUrl})` }}
                title="Displacement map"
              />
            )}
            <div className="tag-label-group">
              <span className="tag-label">{tag.label}</span>
              {s && (
                <span className={`tag-status tag-status-${s.phase}`}>
                  {s.message}
                </span>
              )}
            </div>
            <button
              className="tag-generate-btn"
              disabled={isActive}
              onClick={() => generateTexture(tag)}
            >
              {isActive ? '...' : 'Gen'}
            </button>
          </div>
        );
      })}

      {/* Texture preview lightbox */}
      {preview && (
        <div className="texture-preview-overlay" onClick={() => setPreview(null)}>
          <div className="texture-preview-card" onClick={(e) => e.stopPropagation()}>
            <div className="texture-preview-images">
              {/* Color texture */}
              <div className="texture-preview-main">
                <img src={preview.url} alt={preview.label} />
                <span className="texture-preview-badge">Color</span>
              </div>

              {/* 3D displacement mesh */}
              {preview.displacementUrl && (
                <div className="texture-preview-main">
                  <div className="texture-preview-3d">
                    <Suspense fallback={<div className="texture-preview-loading">Loading 3D...</div>}>
                      <DisplacementPreview
                        textureUrl={preview.url}
                        displacementUrl={preview.displacementUrl}
                        displacementScale={dispScale}
                      />
                    </Suspense>
                  </div>
                  <span className="texture-preview-badge">Displacement 3D</span>
                </div>
              )}
            </div>

            {/* Displacement slider */}
            {preview.displacementUrl && (
              <div className="texture-preview-slider-row">
                <label className="texture-preview-slider-label">Displacement</label>
                <input
                  className="texture-preview-slider"
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={dispScale}
                  onChange={(e) => setDispScale(parseFloat(e.target.value))}
                />
                <span className="texture-preview-slider-value">{dispScale.toFixed(2)}</span>
              </div>
            )}

            <div className="texture-preview-label">{preview.label}</div>
            <button className="texture-preview-close" onClick={() => setPreview(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
