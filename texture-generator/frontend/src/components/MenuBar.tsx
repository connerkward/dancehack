import { useState, useEffect, useRef } from 'react';

export default function MenuBar({
  onExportProject,
  onImportProject,
  onExportMesh,
  onExportUSDZ,
  onExportOBJ,
}: {
  onExportProject: () => void;
  onImportProject: () => void;
  onExportMesh: () => void;
  onExportUSDZ: () => void;
  onExportOBJ: () => void;
}) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handle = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, [open]);

  return (
    <div className="menu-bar">
      <div className="menu-item" ref={menuRef}>
        <button
          className={`menu-item-btn${open ? ' active' : ''}`}
          onClick={() => setOpen(!open)}
        >
          File
        </button>
        {open && (
          <div className="menu-dropdown">
            <button className="menu-dropdown-item" onClick={() => { onExportProject(); setOpen(false); }}>
              Export Project
            </button>
            <button className="menu-dropdown-item" onClick={() => { onImportProject(); setOpen(false); }}>
              Import Project
            </button>
            <div className="menu-dropdown-divider" />
            <button className="menu-dropdown-item" onClick={() => { onExportMesh(); setOpen(false); }}>
              Export Mesh (.glb)
            </button>
            <button className="menu-dropdown-item" onClick={() => { onExportUSDZ(); setOpen(false); }}>
              Export Mesh (.usdz)
            </button>
            <button className="menu-dropdown-item" onClick={() => { onExportOBJ(); setOpen(false); }}>
              Export Mesh (.obj)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
