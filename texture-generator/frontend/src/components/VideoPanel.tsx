import { useRef, useEffect } from 'react';

export default function VideoPanel({
  currentTime,
  isPlaying,
  videoUrl,
}: {
  currentTime: number;
  isPlaying: boolean;
  videoUrl?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    if (Math.abs(vid.currentTime - currentTime) > 0.15) {
      vid.currentTime = currentTime;
    }
  }, [currentTime]);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    if (isPlaying) vid.play().catch(() => {});
    else vid.pause();
  }, [isPlaying]);

  return (
    <div className="video-panel">
      {videoUrl ? (
        <video
          ref={videoRef}
          src={videoUrl}
          muted
          playsInline
          className="video-element"
        />
      ) : (
        <div className="video-placeholder">
          <div className="video-placeholder-icon">&#9654;</div>
          <div className="video-placeholder-text">Dancer Video</div>
          <div className="video-placeholder-hint">Drop video file or set URL</div>
        </div>
      )}
    </div>
  );
}
