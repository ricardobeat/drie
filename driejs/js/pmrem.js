// three's PMREMGenerator for driejs. fromScene() mirrors the scene into drie and has drie
// capture it in every direction from `position`, in linear light (drie's
// Renderer.capture_environment with hdr), at the next render: the texture it returns is
// drie's environment map, for scene.environment or a material's envMap. drie blurs the map
// for rough surfaces itself, so `sigma` is not needed.
//
// Not here: fromEquirectangular and fromCubemap.
import { Texture, Vector3 } from "../vendor/three.core.js";

const N = globalThis.__drie;

export class PMREMGenerator {
	constructor(renderer) {
		this._renderer = renderer;
	}

	fromScene(scene, sigma = 0, near = 0.1, far = 100, options = {}) {
		const position = options.position ?? new Vector3();
		let sceneId;
		if (scene.isScene) {
			sceneId = this._renderer._mirror(scene);
		} else if (scene.isSky) {
			// a Sky on its own (three renders any object as a scene); a drie object has one
			// scene, and this one may be in another already, so the capture gets its own sky
			const N = globalThis.__drie;
			sceneId = N.scene();
			const sky = N.sky();
			N.add(sceneId, sky);
			N.skyParams(sky, ...this._renderer._skyArgs(scene));
		} else {
			throw new Error("PMREMGenerator.fromScene: only a Scene or a Sky can be captured in driejs");
		}
		const texture = new Texture();
		texture.isPMREMTexture = true;
		texture.__drieEnvironment = N.environment(sceneId, position.x, position.y, position.z, near, far);
		return { texture, isWebGLRenderTarget: true, width: 1024, height: 512, dispose() {} };
	}

	fromEquirectangular() {
		throw new Error("driejs: PMREMGenerator.fromEquirectangular is not supported");
	}

	fromCubemap() {
		throw new Error("driejs: PMREMGenerator.fromCubemap is not supported");
	}

	compileCubemapShader() {}
	compileEquirectangularShader() {}
	dispose() {}
}
