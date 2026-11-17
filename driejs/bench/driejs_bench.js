// The transform cases of three_bench.js in drie-js (../drie-js): its three.js-shaped JS API
// keeps position, rotation and scale in JS and drie computes the matrices in C. Each frame
// does what drie-js's `renderer.render()` does before drawing: push every object's
// transform and visibility to drie (the module's `pushTransform`, repeated here because it is
// not exported), then drie's update_matrix_world. Culling and sorting happen in C inside
// drie's render and are not measured. Run from ../drie-js:
//
//   out/drie_js --script ../driejs/bench/driejs_bench.js
import * as THREE from "three";

const N = globalThis.__drie;
const now = () => N.now();
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

// js/three.js's pushTransform
function pushTransform(object) {
	if (object._id === null) return;
	N.transform(
		object._id,
		object.position.x, object.position.y, object.position.z,
		object.rotation.x, object.rotation.y, object.rotation.z,
		object.scale.x, object.scale.y, object.scale.z,
	);
	N.visible(object._id, !!object.visible);
	if (object._lookAt) N.lookAt(object._id, object._lookAt.x, object._lookAt.y, object._lookAt.z);
	for (const child of object.children) pushTransform(child);
}

function makeScene(n) {
	const scene = new THREE.Scene();
	const geometry = new THREE.BoxGeometry(1, 1, 1);
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
			m.scale.set(0.3, 0.3, 0.3);
			group.add(m);
			meshes.push(m);
		}
	}
	return { scene, meshes };
}

for (const n of [1000, 5000, 20000]) {
	const { scene, meshes } = makeScene(n);
	pushTransform(scene);
	N.updateMatrixWorld(scene._id);

	report("transforms: all moving", n, time((f) => {
		for (let i = 0; i < meshes.length; i++) meshes[i].rotation.y = f * 0.01 + i;
		pushTransform(scene);
		N.updateMatrixWorld(scene._id);
	}));

	report("transforms: nothing moving", n, time(() => {
		pushTransform(scene);
		N.updateMatrixWorld(scene._id);
	}));

	report("  of which pushTransform", n, time(() => pushTransform(scene)));
	report("  of which updateMatrixWorld (C)", n, time(() => N.updateMatrixWorld(scene._id)));
}
