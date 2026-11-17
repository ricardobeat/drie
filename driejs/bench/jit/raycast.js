// Raycaster.intersectObject against a sphere of 8064 triangles: Mesh.raycast,
// Ray.intersectTriangle per triangle, BufferAttribute reads (fromBufferAttribute / getX).
import { THREE, run } from "./common.js";
const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), new THREE.MeshBasicMaterial());
sphere.updateMatrixWorld();
const ray = new THREE.Raycaster();
const hits = [];
run("raycast 8k triangles", 300, (f) => {
	ray.set(new THREE.Vector3(Math.sin(f) * 0.5, 0, 5), new THREE.Vector3(0, 0, -1));
	hits.length = 0;
	ray.intersectObject(sphere, false, hits);
});
