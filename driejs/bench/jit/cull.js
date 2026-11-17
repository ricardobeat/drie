// WebGLRenderer's CPU work before drawing: updateMatrixWorld, frustum test per mesh
// (traverseVisible callback, Frustum.intersectsObject -> Sphere.applyMatrix4), depth sort.
import { THREE, run, makeScene } from "./common.js";
const { scene, camera } = makeScene(5000);
const frustum = new THREE.Frustum();
const proj = new THREE.Matrix4();
const v = new THREE.Vector3();
const list = [];
run("cull + sort 5000", 100, () => {
	scene.updateMatrixWorld();
	camera.updateMatrixWorld();
	proj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
	frustum.setFromProjectionMatrix(proj);
	list.length = 0;
	scene.traverseVisible((o) => {
		if (!o.isMesh) return;
		if (o.frustumCulled && !frustum.intersectsObject(o)) return;
		v.setFromMatrixPosition(o.matrixWorld).applyMatrix4(proj);
		list.push({ object: o, z: v.z });
	});
	list.sort((a, b) => a.z - b.z);
});
