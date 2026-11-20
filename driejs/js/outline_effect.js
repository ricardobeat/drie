// three's OutlineEffect for driejs (three/addons/effects/OutlineEffect.js). three draws the
// scene, then draws it again with every mesh's material swapped for a shader that pushes its
// back faces out along their normals. Here render() draws the scene once, with each mesh's
// outline beside it: a drie mesh of the same geometry, its back faces drawn in the outline's
// color with drie's Material.outline pushing them out (see DrieRenderer._outlineOf). A mesh's
// material.userData.outlineParameters (thickness, color, alpha, visible) are read every frame.
import { Color } from "../vendor/three.core.js";

export class OutlineEffect {
	constructor(renderer, parameters = {}) {
		this.enabled = true;
		this.autoClear = parameters.autoClear !== undefined ? parameters.autoClear : true;
		this.domElement = renderer.domElement;
		this.shadowMap = renderer.shadowMap;
		this.defaultThickness = parameters.defaultThickness !== undefined ? parameters.defaultThickness : 0.003;
		this.defaultColor = new Color().fromArray(parameters.defaultColor !== undefined ? parameters.defaultColor : [0, 0, 0]);
		this.defaultAlpha = parameters.defaultAlpha !== undefined ? parameters.defaultAlpha : 1.0;
		this._renderer = renderer;
	}

	render(scene, camera) {
		const renderer = this._renderer;
		renderer._outline = this.enabled ? this : null;
		renderer.render(scene, camera);
		renderer._outline = null;
	}

	// (render() draws the outlines with the scene)
	renderOutline() {}

	setSize(width, height) {
		this._renderer.setSize(width, height);
	}

	setPixelRatio(ratio) {
		this._renderer.setPixelRatio(ratio);
	}

	getPixelRatio() {
		return this._renderer.getPixelRatio();
	}

	getSize(target) {
		return this._renderer.getSize(target);
	}

	setClearColor(color) {
		this._renderer.setClearColor(color);
	}

	setAnimationLoop(callback) {
		this._renderer.setAnimationLoop(callback);
	}

	compile() {}
	dispose() {}
}
