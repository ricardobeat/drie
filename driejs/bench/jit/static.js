// updateMatrixWorld with nothing changed: the same matrix work as moving.js without the
// rotation writes. --opt does well here; kept for comparison.
import { run, makeScene } from "./common.js";
const { scene } = makeScene(5000);
run("static 5000", 200, () => scene.updateMatrixWorld());
