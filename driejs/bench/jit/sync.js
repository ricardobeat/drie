// Copying every mesh's matrixWorld.elements (a 16-element Float32Array) into one big
// Float32Array with TypedArray.prototype.set. No three.js code runs per frame.
import { run, makeScene } from "./common.js";
const { meshes } = makeScene(5000);
const out = new Float32Array(meshes.length * 16);
run("typed array set 5000", 300, () => {
	for (let i = 0; i < meshes.length; i++) out.set(meshes[i].matrixWorld.elements, i * 16);
});
