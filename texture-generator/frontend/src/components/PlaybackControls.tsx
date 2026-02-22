export default function PlaybackControls({
  isPlaying,
  currentTime,
  duration,
  onTogglePlay,
  onSeek,
  canUndo = false,
  canRedo = false,
  onUndo,
  onRedo,
}: {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  onTogglePlay: () => void;
  onSeek: (t: number) => void;
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
}) {
  return (
    <div className="playback-controls">
      <button className="play-btn" onClick={onTogglePlay}>
        {isPlaying ? '||' : '\u25B6'}
      </button>
      <span className="time-display">
        {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
      </span>
      <input
        className="scrubber"
        type="range"
        min={0}
        max={duration}
        step={0.01}
        value={currentTime}
        onChange={(e) => onSeek(parseFloat(e.target.value))}
      />
      <button className="play-btn" onClick={() => onSeek(0)}>
        {'\u23EE'}
      </button>
      <div className="undo-redo-group">
        <button
          className="play-btn undo-btn"
          disabled={!canUndo}
          onClick={onUndo}
          title="Undo (Cmd+Z)"
        >
          {'\u21A9'}
        </button>
        <button
          className="play-btn redo-btn"
          disabled={!canRedo}
          onClick={onRedo}
          title="Redo (Cmd+Shift+Z)"
        >
          {'\u21AA'}
        </button>
      </div>
    </div>
  );
}
