// three's LightProbeGridWebGL for driejs: a drie irradiance volume (IrradianceVolume) over the
// grid's box with its probes, baked by drie (Renderer.bake_irradiance) at the next render:
// each probe's surroundings drawn into six small faces, and the light they bring to a surface
// facing each of the six axes. drie keeps those six values per probe (an ambient cube) where
// three keeps L2 spherical harmonics; materials take the light between the probes around
// them, half a probe's spacing off their surface, on top of the scene's other lights, as
// three's do. While the grid is visible in the scene, it lights what is inside its box, up to
// four grids, the one with the highest `priority` in front where they overlap (drie fades
// each one in over its outermost half cell, where three's switches at its edge). Not done:
// baking a range of probes (`start`, `count`, `pass`).
//
// LightProbeGridHelperWebGL draws nothing here.
import { Box3, Light, Object3D, Vector3 } from "../vendor/three.core.js";

const N = globalThis.__drie;
const _size = new Vector3();

export class LightProbeGridWebGL extends Light {
	constructor(width = 1, height = 1, depth = 1, widthProbes, heightProbes, depthProbes) {
		super(0xffffff, 1);
		this.isLightProbeGrid = true;
		this.type = "LightProbeGrid";
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.resolution = new Vector3(
			widthProbes ?? Math.max(1, Math.round(width)),
			heightProbes ?? Math.max(1, Math.round(height)),
			depthProbes ?? Math.max(1, Math.round(depth))
		);
		this.boundingBox = new Box3();
		this.priority = 0;
		this.texture = null;
		this.updateBoundingBox();
	}

	getProbePosition(ix, iy, iz, target) {
		const p = this.position;
		const r = this.resolution;
		return target.set(
			p.x - this.width / 2 + ((ix + 0.5) * this.width) / r.x,
			p.y - this.height / 2 + ((iy + 0.5) * this.height) / r.y,
			p.z - this.depth / 2 + ((iz + 0.5) * this.depth) / r.z
		);
	}

	updateBoundingBox() {
		_size.set(this.width, this.height, this.depth);
		this.boundingBox.setFromCenterAndSize(this.position, _size);
	}

	bake(renderer, scene, options = {}) {
		const { cubemapSize = 8, near = 0.1, far = 100, bounces = 0 } = options;
		this.updateBoundingBox();
		const sceneId = renderer._mirror(scene);
		const { min, max } = this.boundingBox;
		const r = this.resolution;
		this.__drieVolume = N.irradianceGrid(min.x, min.y, min.z, max.x, max.y, max.z, r.x, r.y, r.z);
		N.bakeIrradiance(this.__drieVolume, sceneId, cubemapSize, near, far, bounces);
		// (three's grid has its probes in a texture once baked)
		this.texture = this.__drieVolume;
	}

	dispose() {}
}

export class LightProbeGridHelperWebGL extends Object3D {
	constructor(probes) {
		super();
		this.probes = probes;
		this.type = "LightProbeGridHelper";
	}

	update() {}

	dispose() {}
}
