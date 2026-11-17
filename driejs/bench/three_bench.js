// Per-frame CPU cost of three.js's own scene work, for running three.js in boomkat with drie
// as the renderer. Runs under boomkat, node and bun:
//
//   boomkat bench/three_bench.js
//   node bench/three_bench.js            (V8 with its JIT)
//   node --jitless bench/three_bench.js  (V8's interpreter only)
//
// Each case is timed over as many frames as fit in about a second (at least 3), after
// warm-up frames, and printed as milliseconds per frame.
import * as THREE from "../vendor/three.core.js";

const now = globalThis.performance?.now ? () => performance.now() : () => Date.now();
const BUDGET_MS = 1000;

function time(frame) {
	for (let i = 0; i < 2; i++) frame(i);
	let frames = 0;
	const start = now();
	let elapsed = 0;
	while (frames < 3 || elapsed < BUDGET_MS) {
		frame(frames + 2);
		frames++;
		elapsed = now() - start;
	}
	return elapsed / frames;
}

function report(name, n, ms) {
	console.log(`${name.padEnd(34)} ${String(n).padStart(6)}  ${ms.toFixed(3).padStart(9)} ms`);
}

// N meshes in groups of 10, like the parts of things placed around a level.
function makeScene(n) {
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

// What WebGLRenderer.render does on the CPU before drawing: update the camera, build the
// frustum, walk the visible tree, cull by bounding sphere, sort by depth.
const _frustum = new THREE.Frustum();
const _proj = new THREE.Matrix4();
const _v = new THREE.Vector3();
function renderList(scene, camera, list) {
	scene.updateMatrixWorld();
	camera.updateMatrixWorld();
	_proj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
	_frustum.setFromProjectionMatrix(_proj);
	list.length = 0;
	scene.traverseVisible((o) => {
		if (!o.isMesh) return;
		if (o.frustumCulled && !_frustum.intersectsObject(o)) return;
		_v.setFromMatrixPosition(o.matrixWorld).applyMatrix4(_proj);
		list.push({ object: o, z: _v.z });
	});
	list.sort((a, b) => a.z - b.z);
}

for (const n of [1000, 5000, 20000]) {
	const { scene, meshes, camera } = makeScene(n);

	report("transforms: all moving", n, time((f) => {
		for (let i = 0; i < meshes.length; i++) meshes[i].rotation.y = f * 0.01 + i;
		scene.updateMatrixWorld();
	}));

	report("transforms: nothing moving", n, time(() => scene.updateMatrixWorld()));

	// a world that stands still: matrixAutoUpdate off everywhere. three still walks the whole
	// tree (matrixWorldAutoUpdate = false only skips an object's own matrix, not its children)
	scene.traverse((o) => { o.matrixAutoUpdate = false; });
	report("transforms: frozen (no autoupdate)", n, time(() => scene.updateMatrixWorld()));
	scene.traverse((o) => { o.matrixAutoUpdate = true; });

	const list = [];
	report("render list: cull + sort", n, time(() => renderList(scene, camera, list)));
	console.log(`${"".padEnd(34)} (${list.length} of ${n} in view)`);

	// handing every mesh's world matrix to native code
	const out = new Float32Array(n * 16);
	report("sync: copy matrixWorld out", n, time(() => {
		for (let i = 0; i < meshes.length; i++) out.set(meshes[i].matrixWorld.elements, i * 16);
	}));
}

// Skinned characters: AnimationMixer sampling a clip on every bone, then bone matrices.
function makeCharacter(bones) {
	const list = [];
	for (let i = 0; i < bones; i++) {
		const b = new THREE.Bone();
		b.position.y = i ? 0.1 : 0;
		if (i) list[(i - 1) >> 1].add(b);
		list.push(b);
	}
	const skeleton = new THREE.Skeleton(list);
	const times = [];
	for (let k = 0; k < 30; k++) times.push(k / 30);
	const tracks = [];
	for (const b of list) {
		const q = [], p = [];
		const quat = new THREE.Quaternion();
		for (let k = 0; k < times.length; k++) {
			quat.setFromEuler(new THREE.Euler(Math.sin(k * 0.3), Math.cos(k * 0.2), 0));
			q.push(quat.x, quat.y, quat.z, quat.w);
			p.push(0, 0.1 + 0.01 * Math.sin(k), 0);
		}
		tracks.push(new THREE.QuaternionKeyframeTrack(`${b.uuid}.quaternion`, times, q));
		tracks.push(new THREE.VectorKeyframeTrack(`${b.uuid}.position`, times, p));
	}
	const root = list[0];
	const mixer = new THREE.AnimationMixer(root);
	mixer.clipAction(new THREE.AnimationClip("walk", 1, tracks)).play();
	return { root, skeleton, mixer };
}

for (const count of [10, 50]) {
	const chars = [];
	for (let i = 0; i < count; i++) chars.push(makeCharacter(60));
	report("animation: 60-bone characters", count, time(() => {
		for (const c of chars) {
			c.mixer.update(1 / 60);
			c.root.updateMatrixWorld();
			c.skeleton.update();
		}
	}));
}

// One ray against a sphere of 64x64 segments (8064 triangles), three's per-triangle test.
{
	const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), new THREE.MeshBasicMaterial());
	sphere.updateMatrixWorld();
	const ray = new THREE.Raycaster();
	const hits = [];
	let found = 0;
	report("raycast: 8k-triangle sphere", 1, time((f) => {
		ray.set(new THREE.Vector3(Math.sin(f) * 0.5, 0, 5), new THREE.Vector3(0, 0, -1));
		hits.length = 0;
		ray.intersectObject(sphere, false, hits);
		found += hits.length;
	}));
}
