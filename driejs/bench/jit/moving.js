// Every mesh's rotation.y changes, then updateMatrixWorld. Writing rotation.y runs Euler's
// setter and its onChange callback (Quaternion.setFromEuler); updateMatrixWorld recomputes
// each matrix (compose, multiplyMatrices).
import { run, makeScene } from "./common.js";
const { scene, meshes } = makeScene(5000);
run("moving 5000", 200, (f) => {
	for (let i = 0; i < meshes.length; i++) meshes[i].rotation.y = f * 0.01 + i;
	scene.updateMatrixWorld();
});
