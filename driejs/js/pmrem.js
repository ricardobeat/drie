// three's PMREMGenerator for driejs. fromScene() mirrors the scene into drie and has drie
// capture it in every direction from `position`, in linear light (drie's
// Renderer.capture_environment with hdr), at the next render: the texture it returns is
// drie's environment map, for scene.environment or a material's envMap. drie blurs the map
// for rough surfaces itself, so `sigma` is not needed.
//
// fromEquirectangular and fromCubemap give the image itself as drie's environment map (a cube
// map as drie's equirectangular map of it), which drie blurs the same way.
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

	fromEquirectangular(equirectangular) {
		return this._of(equirectangular, true);
	}

	fromCubemap(cubemap) {
		return this._of(cubemap, false);
	}

	// A render target whose texture is `source` as drie's environment map (uploaded when first
	// used: the image may still be loading).
	_of(source, equirectangular) {
		const renderer = this._renderer;
		const texture = new Texture();
		texture.isPMREMTexture = true;
		Object.defineProperty(texture, "__drieEnvironment", {
			get: () => renderer._texture(source, equirectangular),
		});
		return { texture, isWebGLRenderTarget: true, width: 1024, height: 512, dispose() {} };
	}

	compileCubemapShader() {}
	compileEquirectangularShader() {}
	dispose() {}
}
