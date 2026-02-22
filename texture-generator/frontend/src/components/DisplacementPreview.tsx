import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

function DisplacedPlane({
  textureUrl,
  displacementUrl,
  displacementScale,
}: {
  textureUrl: string;
  displacementUrl: string;
  displacementScale: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  const colorMap = useMemo(() => {
    const tex = new THREE.TextureLoader().load(textureUrl);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }, [textureUrl]);

  const dispMap = useMemo(() => {
    const tex = new THREE.TextureLoader().load(displacementUrl);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }, [displacementUrl]);

  // Slow auto-rotate
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.z += delta * 0.15;
    }
  });

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2.5, 0, 0]}>
      <planeGeometry args={[2, 2, 256, 256]} />
      <meshStandardMaterial
        map={colorMap}
        displacementMap={dispMap}
        displacementScale={displacementScale}
        side={THREE.DoubleSide}
        metalness={0.2}
        roughness={0.6}
      />
    </mesh>
  );
}

export default function DisplacementPreview({
  textureUrl,
  displacementUrl,
  displacementScale,
}: {
  textureUrl: string;
  displacementUrl: string;
  displacementScale: number;
}) {
  return (
    <Canvas camera={{ position: [0, 1.5, 1.5], fov: 45 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[3, 4, 2]} intensity={1} />
      <directionalLight position={[-2, 3, -1]} intensity={0.3} />
      <DisplacedPlane
        textureUrl={textureUrl}
        displacementUrl={displacementUrl}
        displacementScale={displacementScale}
      />
      <OrbitControls makeDefault enablePan={false} />
    </Canvas>
  );
}
