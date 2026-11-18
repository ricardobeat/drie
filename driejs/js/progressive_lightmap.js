// three's examples/jsm/misc/ProgressiveLightMap.js for driejs, baked by drie's own
// ProgressiveLightMap (lightmap.c3): the same API, and three's packing of the objects' uv1
// (potpack, a margin of three texels). The host maps "three/addons/misc/ProgressiveLightMap.js"
// here.
//
// three's moves the objects into a scene of its own for each update and draws them there with
// a shader of its own, into its render targets. Here the lights go into that scene, as in
// three's, and it is mirrored into drie like any scene; the objects stay where they are (a
// drie object is in one scene), and drie draws them from their place in the main scene. Their
// drie handles exist once the main scene has been rendered: until then update() bakes
// nothing. The light maps are drie's textures (`__drieNative`), made at the first bake.
import { DoubleSide, Mesh, MeshBasicMaterial, PlaneGeometry, Scene, Texture } from "../vendor/three.core.js";
import { potpack } from "../vendor/jsm/libs/potpack.module.js";

const N = globalThis.__drie;

export class ProgressiveLightMap {
	constructor(renderer, res = 1024) {
		this.renderer = renderer;
		this.res = res;
		this.lightMapContainers = [];
		this.scene = new Scene();
		this.firstUpdate = true;
		this.labelMesh = null;
		// (as three's render targets, for their textures: drie's maps 0 and 1)
		this.progressiveLightMap1 = { texture: new Texture(), dispose() {} };
		this.progressiveLightMap2 = { texture: new Texture(), dispose() {} };
		this.progressiveLightMap2.texture.channel = 1;
		// drie's light map, and the object handles it was last given
		this._native = undefined;
		this._objects = "";
	}

	// Sets these objects' materials' light maps and their uv1, and keeps the lights among them.
	addObjectsToLightMap(objects) {
		const boxes = [];
		const padding = 3 / this.res;
		for (let ob = 0; ob < objects.length; ob++) {
			const object = objects[ob];
			if (object.isLight) {
				this.scene.attach(object);
				continue;
			}
			if (object.geometry.hasAttribute("uv") === false) {
				console.warn("THREE.ProgressiveLightMap: All lightmap objects need uvs.");
				continue;
			}
			if (object.geometry.hasAttribute("normal") === false) {
				console.warn("THREE.ProgressiveLightMap: All lightmap objects need normals.");
				continue;
			}
			object.material.lightMap = this.progressiveLightMap2.texture;
			object.material.dithering = true;
			object.castShadow = true;
			object.receiveShadow = true;
			object.renderOrder = 1000 + ob;
			boxes.push({ w: 1 + padding * 2, h: 1 + padding * 2, index: ob });
			this.lightMapContainers.push({ basicMat: object.material, object });
		}
		// the objects' uv1 in one space, each in a square of its own
		const dimensions = potpack(boxes);
		for (const box of boxes) {
			const geometry = objects[box.index].geometry;
			const uv1 = geometry.getAttribute("uv").clone();
			for (let i = 0; i < uv1.array.length; i += uv1.itemSize) {
				uv1.array[i] = (uv1.array[i] + box.x + padding) / dimensions.w;
				uv1.array[i + 1] = (uv1.array[i + 1] + box.y + padding) / dimensions.h;
			}
			geometry.setAttribute("uv1", uv1);
			geometry.getAttribute("uv1").needsUpdate = true;
		}
	}

	// Bakes one more sample of the lights, as they are now, into the light map.
	update(camera, blendWindow = 100, blurEdges = true) {
		if (this.lightMapContainers.length === 0) return;
		this.firstUpdate = false;
		// (each object once: three's adds them again with every call of addObjectsToLightMap)
		const ids = [];
		for (const { object } of this.lightMapContainers) {
			const id = object.__drie;
			if (id === undefined || camera.__drie === undefined) return;
			if (!ids.includes(id)) ids.push(id);
		}
		const renderer = this.renderer;
		const sceneId = renderer._mirror(this.scene);
		if (this._native === undefined) {
			this._native = N.lightmap(renderer._id, this.res, sceneId);
			this.progressiveLightMap1.texture.__drieNative = N.lightmapTexture(this._native, 0);
			this.progressiveLightMap2.texture.__drieNative = N.lightmapTexture(this._native, 1);
		}
		// (driejs makes a new drie mesh when an object's geometry changes, as setting uv1 does)
		const key = ids.join(" ");
		if (this._objects !== key) {
			this._objects = key;
			N.lightmapObjects(this._native, new Float32Array(ids), ids.length);
		}
		N.lightmapUpdate(this._native, camera.__drie, blendWindow, blurEdges);
	}

	// Shows the light map on a plane next to the first object (in its parent, a three Mesh
	// mirrored like any other).
	showDebugLightmap(visible, position = undefined) {
		if (this.lightMapContainers.length === 0) {
			console.warn("THREE.ProgressiveLightMap: Call .showDebugLightmap() after adding the objects.");
			return;
		}
		if (this.labelMesh === null) {
			const labelMaterial = new MeshBasicMaterial({ map: this.progressiveLightMap1.texture, side: DoubleSide });
			this.labelMesh = new Mesh(new PlaneGeometry(100, 100), labelMaterial);
			this.labelMesh.position.y = 250;
			this.lightMapContainers[0].object.parent.add(this.labelMesh);
		}
		if (position !== undefined) this.labelMesh.position.copy(position);
		this.labelMesh.visible = visible;
	}

	dispose() {
		if (this.labelMesh !== null) {
			this.labelMesh.geometry.dispose();
			this.labelMesh.material.dispose();
		}
	}
}
