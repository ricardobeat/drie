// Shared by the per-case scripts in this directory: each isolates one part of three_bench.js
// for boomkat's JIT logs (`boomkat --opt --opt-log <case>.js`).
import * as THREE from "../../vendor/three.core.js";
export { THREE };

const now = globalThis.performance?.now ? () => performance.now() : () => Date.now();

// Runs `frame` for `frames` frames after 20 warm-up frames and prints ms per frame.
export function run(name, frames, frame) {
	for (let i = 0; i < 20; i++) frame(i);
	const start = now();
	for (let i = 0; i < frames; i++) frame(i + 20);
	console.log(`${name}: ${((now() - start) / frames).toFixed(3)} ms/frame`);
}

// N meshes in groups of 10 (three_bench.js's scene).
export function makeScene(n) {
	const scene = new THREE.Scene();
	const geometry = new THREE.BoxGeometry(1, 1, 1);
	geometry.computeBoundingSphere();
	const material = new THREE.MeshStandardMaterial();
	const meshes = [];
	const side = Math.ceil(Math.sqrt(n / 10));
	for (let g = 0; g * 10 < n; g++) {
		const group = new THREE.Group();
		group.position.set((g % side) * 4 - side * 2, 0, Math.floor(g / side) * 4 - side * 2);
		scene.add(group);
		for (let i = 0; i < 10 && g * 10 + i < n; i++) {
			const m = new THREE.Mesh(geometry, material);
			m.position.set((i % 3) - 1, Math.floor(i / 3) * 0.5, 0);
			m.scale.setScalar(0.3);
			group.add(m);
			meshes.push(m);
		}
	}
	const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 500);
	camera.position.set(0, side * 2, side * 3);
	camera.lookAt(0, 0, 0);
	scene.add(camera);
	scene.updateMatrixWorld();
	return { scene, meshes, camera };
}
