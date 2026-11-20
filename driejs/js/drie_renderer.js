// three.js's WebGLRenderer, drawn by drie.
//
// three.js keeps its scene graph and computes the world matrices as usual. render() mirrors
// what it finds into a flat drie scene: each object gets a drie handle the first time it is
// seen (meshes with their geometry and material, lights, cameras; anything else is an
// empty group), then every frame one `sync` call hands drie all world matrices and
// visibilities from a Float32Array, and drie renders. An object on none of the camera's
// layers is hidden (its children are not).
//
// Also mirrored: fog, tone mapping, the colors and numbers of materials and lights (three
// reads them every frame), and shadows: castShadow and receiveShadow, renderer.shadowMap.type
// (VSMShadowMap is drie's VSM, the others its PCF), and the shadows of directional lights
// (each a single box, its shadow.camera's half width: the first is drie's sun shadow, the
// others tiles of its shadow atlas) and spot lights, with their map size, near and far, radius,
// intensity, normalBias, bias, blurSamples and a spot's focus.
// InstancedMesh's matrices, SkinnedMesh (three poses the skeleton, drie moves the vertices),
// Line, LineSegments and LineLoop (a pixel wide, as in WebGL) with LineBasicMaterial and
// LineDashedMaterial, scene.environment from js/pmrem.js's PMREMGenerator or an
// equirectangular texture, scene.background as an equirectangular texture (with
// backgroundBlurriness and backgroundIntensity), and the maps of MeshStandardMaterial and
// MeshPhysicalMaterial: normal, roughness, metalness and ao maps, and the physical material's
// transmission (with its maps), volume, ior, dispersion and specular color and intensity (with
// theirs), clearcoat, sheen, iridescence and anisotropy (with the geometry's tangents). Any material's lightMap (on the first or the
// second uv set: texture.channel) and lightMapIntensity, depthTest and depthWrite. Light maps
// baked by js/progressive_lightmap.js's ProgressiveLightMap.
//
// Not mirrored yet: changes to a material's flags without `needsUpdate`, geometry groups and
// draw ranges, renderOrder, toneMapped, points, sprites, morph targets, point light shadows,
// an emissiveMap that isn't the map, render targets (getRenderTarget() is null,
// setRenderTarget() does nothing) and post-processing.
import { Color, EquirectangularReflectionMapping, EquirectangularRefractionMapping, Matrix4, NearestFilter } from "../vendor/three.core.js";
import { setCanvas } from "./dom.js";

const N = globalThis.__drie;

// handle, visible, 16 numbers of the world matrix: see `sync` in src/main.c3
const STRIDE = 18;

const _color = new Color();
const _bone = new Matrix4();

// The hex of a color brighter than white, scaled down to fit, and how much to scale it back up.
function scaledColor(c) {
	const scale = Math.max(c.r, c.g, c.b, 1);
	_color.setRGB(c.r / scale, c.g / scale, c.b / scale);
	return [_color.getHex(), scale];
}
// With `driejs --profile`: milliseconds spent in each part of render(), summed over the
// frames, and between renders (the app's own code), printed by the host at the end.
const PHASES = ["three.js updateMatrixWorld", "walk + records", "camera, targets, removed", "sync (native)",
	"render (native)", "app code between renders"];
const profile = N.profile ? new Float64Array(PHASES.length) : null;
let profileEnd = 0;
function lap(phase, since) {
	const now = N.now();
	profile[phase] += now - since;
	return now;
}
if (profile) {
	globalThis.__drieReport = (frames) => {
		for (let i = 0; i < PHASES.length; i++) {
			console.log(`  ${PHASES[i].padEnd(28)} ${(profile[i] / frames).toFixed(3).padStart(8)} ms`);
		}
	};
	globalThis.__drieReset = () => profile.fill(0);
}

function materialKind(m) {
	if (m.isMeshNormalMaterial) return 4;
	if (m.isMeshStandardMaterial) return 3;
	if (m.isMeshPhongMaterial) return 2;
	if (m.isMeshLambertMaterial) return 1;
	return 0;
}

function lightKind(l) {
	if (l.isHemisphereLight) return 1;
	if (l.isDirectionalLight) return 2;
	if (l.isPointLight) return 3;
	if (l.isSpotLight) return 4;
	return 0;
}

// An attribute's numbers as a Float32Array, `size` per vertex, or null.
function floats(attribute, size) {
	if (!attribute) return null;
	if (!attribute.isInterleavedBufferAttribute && attribute.array instanceof Float32Array && attribute.itemSize === size) {
		return attribute.array;
	}
	const out = new Float32Array(attribute.count * size);
	for (let i = 0; i < attribute.count; i++) {
		for (let c = 0; c < size; c++) out[i * size + c] = c < attribute.itemSize ? attribute.getComponent(i, c) : 1;
	}
	return out;
}

// Changes when an attribute changes, is added or is removed (computeLineDistances adds one).
function attributeVersions(geometry) {
	let v = geometry.index ? geometry.index.version : 0;
	let count = 0;
	for (const name in geometry.attributes) {
		const a = geometry.attributes[name];
		// (an interleaved attribute has no version of its own: its buffer's)
		v += a.isInterleavedBufferAttribute ? a.data.version : a.version;
		count++;
	}
	return v + count * 1e9;
}

// A line's distances along it (computeLineDistances), as drie keeps them: in the normals' x.
function lineDistanceNormals(attribute) {
	const out = new Float32Array(attribute.count * 3);
	for (let i = 0; i < attribute.count; i++) out[i * 3] = attribute.getX(i);
	return out;
}

export class DrieRenderer {
	constructor(parameters = {}) {
		this.isWebGLRenderer = true;
		this.domElement = document.createElement("canvas");
		setCanvas(this.domElement);
		this.autoClear = true;
		this.sortObjects = true;
		this.shadowMap = { enabled: false, type: 1 };
		this.toneMapping = 0;
		this.toneMappingExposure = 1;
		// the size of the picture transmission materials refract, times the canvas's (three's)
		this.transmissionResolutionScale = 1;
		this.outputColorSpace = "srgb";
		this.info = {
			render: { frame: 0, calls: 0, triangles: 0, points: 0, lines: 0 },
			memory: { geometries: 0, textures: 0 },
			autoReset: true,
			reset() {},
		};
		this._width = window.innerWidth;
		this._height = window.innerHeight;
		this._pixelRatio = 1;
		this._id = N.renderer(parameters.antialias !== false, this._width, this._height, document.title);
		this._records = new Float32Array(1024 * STRIDE);
		this._count = 0;
		this._frame = 0;
		// the state of the scene being mirrored (see _state)
		this._scene = null;
		this._toneMapping = null;
		this._shadowType = null;
	}

	getPixelRatio() {
		return this._pixelRatio;
	}

	setPixelRatio(ratio) {
		this._pixelRatio = ratio;
	}

	getSize(target) {
		return target.set(this._width, this._height);
	}

	setSize(width, height) {
		this._width = width;
		this._height = height;
		N.setSize(this._id, width, height);
	}

	setClearColor(color) {
		N.setClearColor(this._id, _color.set(color).getHex());
	}

	setAnimationLoop(callback) {
		this._loop = callback;
		N.setAnimationLoop(this._id, callback ?? (this._last ? () => this._redraw() : null));
	}

	// A page that renders only when something changes (OrbitControls' "change" event, no
	// animation loop): its last picture is drawn again every frame, so the window has one.
	_redraw() {
		const [scene, camera] = this._last;
		this.render(scene, camera);
	}

	getContext() {
		return null;
	}

	// drie draws the scenes it is given into the window only: no render targets. Addons that
	// put the current one aside and restore it (ProgressiveLightMap) get null, and setting one
	// does nothing (js/progressive_lightmap.js bakes natively instead).
	getRenderTarget() {
		return null;
	}

	setRenderTarget() {}

	compile() {}
	dispose() {}

	render(scene, camera) {
		if (!this._last && !this._loop) N.setAnimationLoop(this._id, () => this._redraw());
		this._last = [scene, camera];
		const sceneId = this._mirror(scene, camera);
		const t = profile ? N.now() : 0;
		N.render(this._id, sceneId, camera.__drie);
		this.info.render.frame++;
		this.info.render.calls = N.info(this._id, 0);
		this.info.render.triangles = N.info(this._id, 1);
		this.info.render.lines = N.info(this._id, 2);
		if (profile) profileEnd = lap(4, t);
	}

	// What drie keeps track of for a scene: its three objects that have a drie handle (to
	// hide the ones that left the scene), how many render() came across this frame, its
	// lights, and [light, drie handle of its target].
	_state(scene) {
		return (scene.__drieState ??= { mirrored: [], visited: 0, lights: [], targets: [] });
	}

	// Brings the scene's drie mirror up to date (with `camera` in it, if given) and returns its
	// drie handle.
	_mirror(scene, camera = null) {
		let t = 0;
		if (profile) {
			t = N.now();
			if (profileEnd) profile[5] += t - profileEnd;
		}
		if (scene.matrixWorldAutoUpdate === true) scene.updateMatrixWorld();
		if (camera && camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
		if (profile) t = lap(0, t);

		const frame = ++this._frame;
		let sceneId = scene.__drie;
		if (sceneId === undefined) sceneId = scene.__drie = N.scene();
		const state = (this._scene = this._state(scene));
		if (scene.background && scene.background.isColor) {
			const hex = scene.background.getHex();
			if (scene.__drieBackground !== hex) N.background(sceneId, (scene.__drieBackground = hex));
		}
		this._fog(scene, sceneId);
		this._background(scene, sceneId);
		const environment = scene.environment;
		const env = environment ? (environment.__drieEnvironment ?? this._texture(environment)) : -1;
		const envIntensity = scene.environmentIntensity ?? 1;
		if (scene.__drieEnvironment !== env || scene.__drieEnvironmentIntensity !== envIntensity) {
			scene.__drieEnvironment = env;
			scene.__drieEnvironmentIntensity = envIntensity;
			N.sceneEnvironment(sceneId, env, envIntensity);
		}
		if (this._toneMapping !== this.toneMapping || this._exposure !== this.toneMappingExposure
			|| this._transmissionScale !== this.transmissionResolutionScale) {
			this._toneMapping = this.toneMapping;
			this._exposure = this.toneMappingExposure;
			this._transmissionScale = this.transmissionResolutionScale;
			N.toneMapping(this._id, this.toneMapping, this.toneMappingExposure, this.transmissionResolutionScale ?? 1);
		}
		if (this._shadowType !== this.shadowMap.type) {
			this._shadowType = this.shadowMap.type;
			N.shadowType(this._id, this.shadowMap.type);
		}

		this._count = 0;
		state.visited = 0;
		// (what the camera sees: an object on none of its layers is hidden, its children not)
		this._layerMask = camera ? camera.layers.mask : 0xffffffff;
		this._walk(scene, true, sceneId, frame);
		if (profile) t = lap(1, t);

		// a camera outside the scene still needs a drie handle in it, or drie would place
		// it from its own (unset) position
		if (camera) {
			if (camera.__drieFrame !== frame) {
				if (camera.__drie === undefined) this._create(camera, sceneId);
				camera.__drieFrame = frame;
				state.visited++;
				this._record(camera.__drie, true, camera.matrixWorld.elements);
			}
			this._project(camera);
		}

		for (const [light, id] of state.targets) this._record(id, true, light.target.matrixWorld.elements);
		for (let i = 0; i < state.lights.length; i++) this._light(state.lights[i]);

		// what was mirrored before and is gone from the scene now (only looked for when fewer
		// objects were seen this frame than have been mirrored)
		const mirrored = state.mirrored;
		if (state.visited !== mirrored.length) {
			for (let i = 0; i < mirrored.length; i++) {
				const o = mirrored[i];
				if (o.__drieFrame !== frame) this._record(o.__drie, false, o.matrixWorld.elements);
			}
		}

		if (profile) t = lap(2, t);
		N.sync(this._records, this._count);
		if (profile) profileEnd = lap(3, t);
		return sceneId;
	}

	_walk(parent, visible, sceneId, frame) {
		const children = parent.children;
		for (let i = 0; i < children.length; i++) {
			const o = children[i];
			const shown = visible && o.visible;
			const drawn = shown && (o.layers.mask & this._layerMask) !== 0;
			let id = o.__drie;
			if (id === undefined) id = this._create(o, sceneId);
			o.__drieFrame = frame;
			this._scene.visited++;
			this._record(id, drawn, o.matrixWorld.elements);
			if (o.isSky) this._sky(o);
			else if (o.isMesh || o.isLine) this._refresh(o, frame, drawn);
			if (o.isInstancedMesh) this._instances(o);
			if (o.isSkinnedMesh && drawn) this._skin(o, frame);
			if (o.children.length !== 0) this._walk(o, shown, sceneId, frame);
		}
	}

	_record(id, visible, elements) {
		let records = this._records;
		const at = this._count * STRIDE;
		if (at + STRIDE > records.length) {
			records = new Float32Array(records.length * 2);
			records.set(this._records);
			this._records = records;
		}
		records[at] = id;
		records[at + 1] = visible ? 1 : 0;
		records.set(elements, at + 2);
		this._count++;
	}

	_create(o, sceneId) {
		let id;
		if (o.isSky) {
			// three's Sky shader, as drie's own Sky (its ShaderMaterial can't be drawn as it is)
			id = N.sky();
		} else if (o.isMesh || o.isLine) {
			id = this._newMesh(o, this._geometry(o.geometry, this._frame), this._material(o.material, this._frame));
			o.__drieGeometry = o.geometry.__drie;
			o.__drieMaterial = (Array.isArray(o.material) ? o.material[0] : o.material).__drie;
		} else if (o.isLight) {
			_color.copy(o.color);
			id = N.light(lightKind(o), _color.getHex(), o.intensity, o.distance ?? 0, o.decay ?? 2, o.angle ?? Math.PI / 3,
				o.penumbra ?? 0, o.groundColor ? o.groundColor.getHex() : 0xffffff);
			if (o.target) {
				const target = N.lightTarget(id);
				if (target !== null) this._scene.targets.push([o, target]);
			}
			this._scene.lights.push(o);
		} else if (o.isPerspectiveCamera) {
			id = N.camera(o.fov, o.aspect, o.near, o.far);
		} else {
			id = N.group();
		}
		N.add(sceneId, id);
		o.__drie = id;
		this._scene.mirrored.push(o);
		return id;
	}

	_newMesh(o, g, m) {
		if (o.isLine) return N.line(g, m, o.isLineSegments ? 1 : o.isLineLoop ? 3 : 2);
		if (o.isInstancedMesh) {
			o.__drieInstances = undefined;
			return N.instancedMesh(g, m, o.count);
		}
		if (o.isSkinnedMesh) return N.skinnedMesh(g, m);
		return N.mesh(g, m);
	}

	// Sky's uniforms, in the order of N.skyParams. (r177's Sky has no clouds: none when its
	// uniforms are missing.)
	_skyArgs(sky) {
		const u = sky.material.uniforms;
		const v = (name, fallback) => (u[name] ? u[name].value : fallback);
		const sun = u.sunPosition.value;
		return [v("turbidity", 2), v("rayleigh", 1), v("mieCoefficient", 0.005), v("mieDirectionalG", 0.8),
			sun.x, sun.y, sun.z, v("cloudScale", 0.0002), v("cloudSpeed", 0.00002), v("cloudCoverage", 0),
			v("cloudDensity", 0.4), v("cloudElevation", 0.5), v("showSunDisc", 1), v("time", 0)];
	}

	// Sky's uniforms, sent when one changed.
	_sky(sky) {
		const args = this._skyArgs(sky);
		const key = args.join(",");
		if (sky.__drieSky === key) return;
		sky.__drieSky = key;
		N.skyParams(sky.__drie, ...args);
	}

	_instances(mesh) {
		const version = mesh.instanceMatrix.version;
		if (mesh.__drieInstances === version && mesh.__drieCount === mesh.count) return;
		mesh.__drieInstances = version;
		mesh.__drieCount = mesh.count;
		N.instances(mesh.__drie, mesh.instanceMatrix.array, mesh.count);
	}

	// Poses a skinned mesh: three's bone matrices (its skeleton is updated once a frame,
	// however many meshes share it), each between the bind matrix and its inverse.
	_skin(mesh, frame) {
		const skeleton = mesh.skeleton;
		if (skeleton.__drieFrame !== frame) {
			skeleton.__drieFrame = frame;
			skeleton.update();
		}
		const bones = skeleton.bones.length;
		let out = mesh.__drieSkin;
		if (!out || out.length !== bones * 16) out = mesh.__drieSkin = new Float32Array(bones * 16);
		for (let i = 0; i < bones; i++) {
			_bone.fromArray(skeleton.boneMatrices, i * 16);
			_bone.premultiply(mesh.bindMatrixInverse).multiply(mesh.bindMatrix);
			_bone.toArray(out, i * 16);
		}
		N.skin(mesh.__drie, out, bones);
	}

	// A mesh whose geometry or material changed (or was replaced) gets a new drie mesh.
	_refresh(mesh, frame, drawn) {
		const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
		const g = this._geometry(mesh.geometry, frame);
		const m = this._material(material, frame);
		if (g === mesh.__drieGeometry && m === mesh.__drieMaterial) {
			this._shadowFlags(mesh);
			return;
		}
		mesh.__drieGeometry = g;
		mesh.__drieMaterial = m;
		// hide the old one: its record is the last one written
		this._records[this._count * STRIDE - STRIDE + 1] = 0;
		const id = this._newMesh(mesh, g, m);
		N.add(this._sceneOf(mesh), id);
		mesh.__drie = id;
		mesh.__drieShadow = undefined;
		this._record(id, drawn, mesh.matrixWorld.elements);
		this._shadowFlags(mesh);
	}

	_shadowFlags(mesh) {
		const flags = (this.shadowMap.enabled && mesh.castShadow ? 2 : 0) | (mesh.receiveShadow ? 1 : 0);
		if (mesh.__drieShadow === flags) return;
		mesh.__drieShadow = flags;
		N.shadow(mesh.__drie, (flags & 2) !== 0, (flags & 1) !== 0);
	}

	_fog(scene, sceneId) {
		const fog = scene.fog;
		let key = 0;
		if (fog) key = `${fog.color.getHex()} ${fog.isFogExp2 ? fog.density : `${fog.near} ${fog.far}`}`;
		if (scene.__drieFog === key) return;
		scene.__drieFog = key;
		if (!fog) N.fog(sceneId, 0, 0, 0, 0, 0);
		else if (fog.isFogExp2) N.fog(sceneId, 2, fog.color.getHex(), 0, 0, fog.density);
		else N.fog(sceneId, 1, fog.color.getHex(), fog.near, fog.far, 0);
	}

	// A light's color, intensity and shadow, sent when they changed.
	_light(light) {
		const v = (light.__drieValues ??= []);
		const ground = light.groundColor;
		if (v[0] !== light.color.r || v[1] !== light.color.g || v[2] !== light.color.b || v[3] !== light.intensity
			|| (ground && (v[4] !== ground.r || v[5] !== ground.g || v[6] !== ground.b))) {
			v[0] = light.color.r;
			v[1] = light.color.g;
			v[2] = light.color.b;
			v[3] = light.intensity;
			if (ground) {
				v[4] = ground.r;
				v[5] = ground.g;
				v[6] = ground.b;
			}
			N.lightUpdate(light.__drie, light.color.getHex(), light.intensity, ground ? ground.getHex() : 0);
		}
		if (!light.isDirectionalLight && !light.isSpotLight) return;
		const shadow = light.shadow;
		const cam = shadow.camera;
		const cast = this.shadowMap.enabled && light.castShadow;
		// (a spot light's shadow covers its cone: no box)
		const extent = light.isDirectionalLight ? Math.max(Math.abs(cam.right - cam.left), Math.abs(cam.top - cam.bottom)) / 2 : 0;
		const focus = shadow.focus ?? 1;
		const key = cast ? `${shadow.mapSize.x} ${shadow.radius} ${cam.near} ${cam.far} ${extent} ${shadow.normalBias} ${shadow.intensity} ${shadow.bias} ${shadow.blurSamples} ${focus}` : "";
		if (light.__drieShadowKey === key) return;
		light.__drieShadowKey = key;
		N.lightShadow(light.__drie, cast, shadow.mapSize.x, shadow.radius, cam.near, cam.far, extent, shadow.normalBias,
			shadow.intensity ?? 1, shadow.bias, shadow.blurSamples ?? 8, focus);
	}

	_sceneOf(o) {
		while (o.parent) o = o.parent;
		return o.__drie;
	}

	// Geometries and materials are looked at once per frame, however many meshes share them.
	_geometry(geometry, frame) {
		if (geometry.__drieFrame === frame) return geometry.__drie;
		geometry.__drieFrame = frame;
		const version = attributeVersions(geometry);
		if (geometry.__drie !== undefined && geometry.__drieVersion === version) return geometry.__drie;
		const a = geometry.attributes;
		const color = a.color ? floats(a.color, a.color.itemSize === 4 ? 4 : 3) : null;
		let joints = null;
		if (a.skinIndex) {
			joints = new Uint16Array(a.skinIndex.count * 4);
			for (let i = 0; i < a.skinIndex.count; i++) {
				for (let c = 0; c < 4; c++) joints[i * 4 + c] = a.skinIndex.getComponent(i, c);
			}
		}
		const normals = a.normal ? floats(a.normal, 3) : a.lineDistance ? lineDistanceNormals(a.lineDistance) : null;
		geometry.__drie = N.geometry(floats(a.position, 3), normals, floats(a.uv, 2), color,
			geometry.index ? geometry.index.array : null, joints, floats(a.skinWeight, 4), floats(a.uv1, 2),
			floats(a.tangent, 4));
		geometry.__drieVersion = version;
		return geometry.__drie;
	}

	// scene.background as an equirectangular texture (a color: see render()).
	_background(scene, sceneId) {
		const background = scene.background;
		// (a PMREMGenerator's texture is drie's capture of a scene)
		const texture = background && background.isTexture ? (background.__drieEnvironment ?? this._texture(background)) : -1;
		const key = texture < 0 ? -1 : `${texture} ${scene.backgroundBlurriness} ${scene.backgroundIntensity}`;
		if (scene.__drieBackgroundTexture === key) return;
		scene.__drieBackgroundTexture = key;
		N.backgroundTexture(sceneId, texture, scene.backgroundBlurriness ?? 0, scene.backgroundIntensity ?? 1);
	}

	// An image drie loads from its file (`path`) or decodes from memory (`data`, from a Blob,
	// as GLTFLoader makes them), or a DataTexture of half floats or floats (an HDR image, as
	// UltraHDRLoader and HDRLoader make them).
	_texture(texture) {
		// (a texture that drie made, as the light map of js/progressive_lightmap.js)
		if (texture?.__drieNative !== undefined) return texture.__drieNative;
		if (texture?.isCubeTexture) return this._cubeTexture(texture);
		const image = texture?.image;
		// (a canvas: the pixels its 2D context drew)
		const canvas = image?.tagName === "CANVAS" ? image.getContext("2d")._buffer() : null;
		const source = canvas ? canvas.data : image ? (image.data ?? image.path) : undefined;
		if (source === undefined) return -1;
		if (texture.__drie !== undefined && texture.__drieVersion === texture.version) return texture.__drie;
		// (three's equirectangular maps have the sky at v = 1, drie's at v = 0: flipped the other way)
		const equirect = texture.mapping === EquirectangularReflectionMapping || texture.mapping === EquirectangularRefractionMapping;
		const flipY = equirect ? !texture.flipY : texture.flipY;
		if (texture.isDataTexture && (source instanceof Uint16Array || source instanceof Float32Array)) {
			if (!image.width || !image.height) return -1;
			texture.__drie = N.floatTexture(source, image.width, image.height, flipY);
		} else {
			// Clones of one image (three's texture.clone(), as GLTFLoader makes for
			// KHR_texture_transform and texCoord) share its upload: drie's Texture.clone
			const images = canvas || !texture.source ? null : (texture.source.__drieImages ??= new Map());
			const key = `${texture.colorSpace === "srgb"} ${flipY} ${texture.source?.version}`;
			const shown = images?.get(key);
			if (shown !== undefined) {
				texture.__drie = N.textureClone(shown, texture.repeat.x, texture.repeat.y, texture.offset.x,
					texture.offset.y, texture.rotation, texture.center.x, texture.center.y, texture.channel ?? 0);
			} else {
				texture.__drie = N.texture(source, texture.colorSpace === "srgb", texture.wrapS, texture.wrapT,
					texture.anisotropy, texture.repeat.x, texture.repeat.y, texture.offset.x, texture.offset.y, flipY,
					texture.rotation, texture.center.x, texture.center.y, canvas ? canvas.width : 0,
					canvas ? canvas.height : 0, texture.magFilter === NearestFilter, texture.channel ?? 0);
				images?.set(key, texture.__drie);
			}
		}
		texture.__drieVersion = texture.version;
		return texture.__drie;
	}

	// A map on the first or the second uv set (three's texture.channel; drie's geometries have
	// only those two).
	_map(texture) {
		return texture && texture.channel > 1 ? -1 : this._texture(texture);
	}

	// A CubeTexture of six DataTextures of half floats or floats (HDRCubeTextureLoader's), as
	// drie's equirectangular map of it.
	_cubeTexture(texture) {
		if (texture.__drie !== undefined && texture.__drieVersion === texture.version) return texture.__drie;
		const faces = texture.images ?? texture.image;
		if (!Array.isArray(faces) || faces.length !== 6) return -1;
		const data = faces.map((f) => f?.image?.data ?? f?.data);
		const size = faces[0]?.image?.width ?? faces[0]?.width;
		if (!size || data.some((d) => !(d instanceof Uint16Array || d instanceof Float32Array))) {
			throw new Error("driejs: only cube maps of half floats or floats (HDRCubeTextureLoader) are supported");
		}
		texture.__drie = N.cubeTexture(...data, size);
		texture.__drieVersion = texture.version;
		return texture.__drie;
	}

	// The maps besides `map`, sent when one of them changes.
	_maps(material) {
		// (GLTFLoader clones metallicRoughnessTexture for each with a KHR_texture_transform: one
		// image is one map for drie, which reads both from its channels only when they are one)
		const rough = material.roughnessMap;
		const metal = rough && material.metalnessMap?.source === rough.source ? rough : material.metalnessMap;
		const maps = [this._map(material.normalMap), material.normalScale ? material.normalScale.x : 1,
			material.normalScale ? material.normalScale.y : 1, this._map(rough), this._map(metal), this._map(material.aoMap),
			material.aoMapIntensity ?? 1, this._map(material.specularColorMap),
			this._map(material.specularIntensityMap), this._map(material.transmissionMap),
			this._map(material.thicknessMap), this._map(material.clearcoatMap),
			this._map(material.clearcoatRoughnessMap), this._map(material.clearcoatNormalMap),
			material.clearcoatNormalScale ? material.clearcoatNormalScale.x : 1,
			material.clearcoatNormalScale ? material.clearcoatNormalScale.y : 1,
			this._map(material.sheenColorMap), this._map(material.sheenRoughnessMap),
			this._map(material.iridescenceMap), this._map(material.iridescenceThicknessMap),
			this._map(material.anisotropyMap), this._map(material.alphaMap),
			// (the color map, when it is the same image: drie's map_emissive)
			material.__drieMapEmissive ? -1 : this._map(material.emissiveMap),
			// (its own surroundings, instead of the scene's environment)
			material.envMap ? (material.envMap.__drieEnvironment ?? this._texture(material.envMap)) : -1,
			material.envMapIntensity ?? 1];
		const key = maps.join(" ");
		if (material.__drieMaps === key) return;
		material.__drieMaps = key;
		N.materialMaps(material.__drie, ...maps);
	}

	// The light map and its intensity (a uniform in three: no needsUpdate), sent when they
	// change.
	_lightMap(material) {
		const map = this._map(material.lightMap);
		const intensity = material.lightMapIntensity ?? 1;
		const key = `${map} ${intensity}`;
		if (material.__drieLightMap === key) return;
		material.__drieLightMap = key;
		N.materialLightMap(material.__drie, map, intensity);
	}

	// depthTest and depthWrite (three sets them for each draw: no needsUpdate either).
	_depth(material) {
		const key = (material.depthTest === false ? 1 : 0) + (material.depthWrite === false ? 2 : 0);
		if (material.__drieDepth === key) return;
		material.__drieDepth = key;
		N.materialDepth(material.__drie, material.depthTest !== false, material.depthWrite !== false);
	}

	// MeshPhysicalMaterial's clear coat, sheen, iridescence and anisotropy, sent when they change.
	_layers(material) {
		if (!material.isMeshPhysicalMaterial) return;
		const sc = material.sheenColor;
		const layers = [material.clearcoat, material.clearcoatRoughness, material.sheen, sc.r, sc.g, sc.b,
			material.sheenRoughness, material.iridescence, material.iridescenceIOR,
			material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum,
			material.anisotropy, material.anisotropyRotation];
		const key = layers.join(" ");
		if (material.__drieLayers === key) return;
		material.__drieLayers = key;
		N.materialLayers(material.__drie, ...layers);
	}

	_material(material, frame) {
		if (Array.isArray(material)) material = material[0];
		if (material.__drieFrame === frame) return material.__drie;
		material.__drieFrame = frame;
		const map = this._texture(material.map);
		if (material.__drie !== undefined && material.__drieVersion === material.version && material.__drieMap === map) {
			this._materialValues(material);
			this._maps(material);
			this._layers(material);
			this._lightMap(material);
			this._depth(material);
			return material.__drie;
		}
		const sc = material.specularColor;
		const [color, colorScale] = material.color ? scaledColor(material.color) : [0xffffff, 1];
		const e = material.emissiveMap;
		const mapEmissive = !!(e && material.map && (e === material.map || e.source === material.map.source));
		material.__drieMapEmissive = mapEmissive;
		material.__drie = N.material(materialKind(material), color,
			material.emissive ? material.emissive.getHex() : 0, material.specular ? material.specular.getHex() : 0x111111,
			material.shininess ?? 30, material.roughness ?? 1, material.metalness ?? 0, material.opacity,
			material.transparent, material.side, material.wireframe === true, map, material.vertexColors,
			material.alphaTest, colorScale, mapEmissive, material.emissiveIntensity ?? 1,
			material.transmission ?? 0, material.thickness ?? 0, material.ior ?? 1.5, material.dispersion ?? 0,
			material.attenuationColor ? material.attenuationColor.getHex() : 0xffffff,
			Number.isFinite(material.attenuationDistance) ? material.attenuationDistance : 0,
			sc ? sc.r : 1, sc ? sc.g : 1, sc ? sc.b : 1, material.specularIntensity ?? 1);
		material.__drieVersion = material.version;
		material.__drieMap = map;
		material.__drieValues = null;
		material.__drieDash = undefined;
		material.__drieMaps = undefined;
		material.__drieLayers = undefined;
		material.__drieLightMap = undefined;
		material.__drieDepth = undefined;
		this._materialValues(material);
		this._maps(material);
		this._layers(material);
		this._lightMap(material);
		this._depth(material);
		return material.__drie;
	}

	// Colors and numbers that three reads into uniforms every frame, so changing them needs
	// no needsUpdate: sent to drie when they differ from the last frame.
	_materialValues(material) {
		if (material.isLineDashedMaterial) {
			const dash = `${material.dashSize} ${material.gapSize} ${material.scale}`;
			if (material.__drieDash !== dash) {
				material.__drieDash = dash;
				N.lineDash(material.__drie, material.dashSize, material.gapSize, material.scale);
			}
		}
		const c = material.color;
		const e = material.emissive;
		const s = material.specular;
		const a = material.attenuationColor;
		const ad = Number.isFinite(material.attenuationDistance) ? material.attenuationDistance : 0;
		const sc = material.specularColor;
		const v = material.__drieValues;
		if (v !== null && v !== undefined && (!c || (v[0] === c.r && v[1] === c.g && v[2] === c.b))
			&& (!e || (v[3] === e.r && v[4] === e.g && v[5] === e.b)) && (!s || (v[6] === s.r && v[7] === s.g && v[8] === s.b))
			&& v[9] === material.opacity && v[10] === material.shininess && v[11] === material.roughness
			&& v[12] === material.metalness && v[13] === material.transmission && v[14] === material.thickness
			&& v[15] === material.ior && v[16] === material.dispersion && (!a || v[17] === a.getHex())
			&& v[18] === ad && (!sc || (v[19] === sc.r && v[20] === sc.g && v[21] === sc.b))
			&& v[22] === material.specularIntensity) {
			return;
		}
		material.__drieValues = [c?.r, c?.g, c?.b, e?.r, e?.g, e?.b, s?.r, s?.g, s?.b, material.opacity, material.shininess,
			material.roughness, material.metalness, material.transmission, material.thickness, material.ior,
			material.dispersion, a?.getHex(), ad, sc?.r, sc?.g, sc?.b, material.specularIntensity];
		if (v === null) return;
		const [color, colorScale] = c ? scaledColor(c) : [0xffffff, 1];
		N.materialUpdate(material.__drie, color, e ? e.getHex() : 0, s ? s.getHex() : 0x111111,
			material.shininess ?? 30, material.roughness ?? 1, material.metalness ?? 0, material.opacity, colorScale,
			material.transmission ?? 0, material.thickness ?? 0, material.ior ?? 1.5, material.dispersion ?? 0,
			a ? a.getHex() : 0xffffff, ad, sc ? sc.r : 1, sc ? sc.g : 1, sc ? sc.b : 1, material.specularIntensity ?? 1);
	}

	_project(camera) {
		const p = camera.__drieProjection ?? (camera.__drieProjection = []);
		if (p[0] === camera.fov && p[1] === camera.aspect && p[2] === camera.near && p[3] === camera.far && p[4] === camera.zoom) return;
		p[0] = camera.fov;
		p[1] = camera.aspect;
		p[2] = camera.near;
		p[3] = camera.far;
		p[4] = camera.zoom;
		N.cameraUpdate(camera.__drie, camera.fov, camera.aspect, camera.near, camera.far, camera.zoom);
	}
}
