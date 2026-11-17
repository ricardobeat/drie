// Only the rotation writes of moving.js: Euler's `y` setter and onChange ->
// Quaternion.setFromEuler, without updateMatrixWorld.
import { run, makeScene } from "./common.js";
const { meshes } = makeScene(5000);
run("rotation writes 5000", 200, (f) => {
	for (let i = 0; i < meshes.length; i++) meshes[i].rotation.y = f * 0.01 + i;
});
