// AnimationMixer on 10 characters of 60 bones (a quaternion and a position track per bone),
// then updateMatrixWorld and Skeleton.update (bone matrices into a Float32Array).
import { THREE, run } from "./common.js";
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
	const mixer = new THREE.AnimationMixer(list[0]);
	mixer.clipAction(new THREE.AnimationClip("walk", 1, tracks)).play();
	return { root: list[0], skeleton, mixer };
}
const chars = [];
for (let i = 0; i < 10; i++) chars.push(makeCharacter(60));
run("animation 10 x 60 bones", 300, () => {
	for (const c of chars) {
		c.mixer.update(1 / 60);
		c.root.updateMatrixWorld();
		c.skeleton.update();
	}
});
