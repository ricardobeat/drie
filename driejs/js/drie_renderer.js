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
// InstancedMesh's matrices and instanceColor, SkinnedMesh (three poses the skeleton, drie
// moves the vertices), Line, LineSegments and LineLoop (a pixel wide, as in WebGL) with
// LineBasicMaterial and LineDashedMaterial, scene.environment from js/pmrem.js's
// PMREMGenerator or an equirectangular texture or a cube map (of images or of HDR
// DataTextures), scene.background as either (with backgroundBlurriness and
// backgroundIntensity), the envMap of the basic, lambert and phong materials (reflectivity,
// combine, refractionRatio and the refraction mappings, specularMap), MeshMatcapMaterial,
// MeshToonMaterial (gradientMap), bumpMap, the outlines of js/outline_effect.js's
// OutlineEffect, DataTextures of bytes (RGBA, RG or red), blending (every mode,
// premultipliedAlpha, and custom blending's equations and factors), toneMapped, a texture's
// repeat and offset as they change, scene.background as a flat texture, displacementMap (with
// its scale and bias), OrthographicCamera, MeshDepthMaterial (depthPacking), setViewport,
// setScissor and setScissorTest, autoClear and clear(), SunLight (drie's directional light
// with cascaded shadows), an equirectangular texture as a map (GroundedSkybox's), and the maps
// of MeshStandardMaterial and MeshPhysicalMaterial: normal, roughness, metalness and ao maps,
// and the physical material's transmission (with its maps), volume, ior, dispersion and
// specular color and intensity (with theirs), clearcoat, sheen, iridescence and anisotropy
// (with the geometry's tangents). Any material's lightMap (on the first or the second uv set:
// texture.channel) and lightMapIntensity, depthTest and depthWrite. Light maps baked by
// js/progressive_lightmap.js's ProgressiveLightMap.
//
// Not mirrored yet: changes to a material's flags without `needsUpdate`, geometry groups and
// draw ranges, renderOrder, points, sprites, morph targets, point light shadows, an
// emissiveMap that isn't the map, render targets (getRenderTarget() is null, setRenderTarget()
// does nothing) and post-processing.
import { BackSide, Color, CubeCamera, CubeRefractionMapping, EquirectangularReflectionMapping, EquirectangularRefractionMapping,
	LightProbe, Matrix4, MeshBasicMaterial, NearestFilter, RedFormat, RGBAFormat, RGFormat, Vector3, Vector4 } from "../vendor/three.core.js";
import { setCanvas } from "./dom.js";

const N = globalThis.__drie;

const uvKey = (t) => `${t.repeat.x} ${t.repeat.y} ${t.offset.x} ${t.offset.y} ${t.rotation} ${t.center.x} ${t.center.y}`;

// A DataTexture's bytes as RGBA (three's RedFormat, RGFormat and RGBAFormat; a red texture
// read as three reads it: its red in r, 0 in g and b), or null for another format.
function rgbaBytes(image, format) {
	const data = image.data;
	const count = image.width * image.height;
	const channels = format === RGBAFormat ? 4 : format === RGFormat ? 2 : format === RedFormat ? 1 : 0;
	if (channels === 0 || data.length < count * channels) return null;
	if (channels === 4) return data;
	const out = new Uint8Array(count * 4);
	for (let i = 0; i < count; i++) {
		out[i * 4] = data[i * channels];
		if (channels === 2) out[i * 4 + 1] = data[i * channels + 1];
		out[i * 4 + 3] = 255;
	}
	return out;
}

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
	if (m.isMeshDepthMaterial) return 7;
	if (m.isMeshToonMaterial) return 6;
	if (m.isMeshMatcapMaterial) return 5;
	if (m.isMeshNormalMaterial) return 4;
	if (m.isMeshStandardMaterial) return 3;
	if (m.isMeshPhongMaterial) return 2;
	if (m.isMeshLambertMaterial) return 1;
	return 0;
}

function lightKind(l) {
	if (l.isHemisphereLight) return 1;
	// (SunLight: a directional light from its position towards the origin, drie's without a
	// target, with cascaded shadows)
	if (l.isDirectionalLight || l.isSunLight) return 2;
	if (l.isPointLight) return 3;
	if (l.isSpotLight) return 4;
	if (l.isRectAreaLight) return 5;
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
		// (three's WebGLCoordinateSystem: clip space z from -1 to 1, which CubeCamera reads)
		this.coordinateSystem = 2000;
		// no WebXR: what three's code reads of it
		this.xr = { enabled: false, isPresenting: false, addEventListener() {}, removeEventListener() {} };
		this.toneMapping = 0;
		this.toneMappingExposure = 1;
		// the size of the picture transmission materials refract, times the canvas's (three's)
		this.transmissionResolutionScale = 1;
		this.outputColorSpace = "srgb";
		// what three's examples ask of the GPU (drie's Metal, D3D11 and GL 4.3 backends all have it)
		this.capabilities = {
			isWebGL2: true,
			precision: "highp",
			logarithmicDepthBuffer: false,
			reversedDepthBuffer: false,
			maxTextures: 16,
			maxVertexTextures: 16,
			maxTextureSize: 16384,
			maxCubemapSize: 16384,
			maxSamples: 4,
			getMaxAnisotropy: () => 16,
			getMaxPrecision: () => "highp",
		};
		// (no WebGL extensions: KTX2Loader transcodes to plain RGBA)
		this.extensions = { has: () => false, get: () => null };
		this.info = {
			render: { frame: 0, calls: 0, triangles: 0, points: 0, lines: 0 },
			memory: { geometries: 0, textures: 0 },
			autoReset: true,
			reset() {},
		};
		this._width = window.innerWidth;
		this._height = window.innerHeight;
		this._pixelRatio = 1;
		this._viewport = new Vector4(0, 0, this._width, this._height);
		this._scissor = new Vector4(0, 0, this._width, this._height);
		this._scissorTest = false;
		this._id = N.renderer(parameters.antialias !== false, this._width, this._height, document.title);
		this._records = new Float32Array(1024 * STRIDE);
		this._count = 0;
		this._frame = 0;
		// the state of the scene being mirrored (see _state)
		this._scene = null;
		this._toneMapping = null;
		this._shadowType = null;
		// the OutlineEffect drawing through this renderer, during its render()
		this._outline = null;
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

	getDrawingBufferSize(target) {
		return target.set(this._width * this._pixelRatio, this._height * this._pixelRatio).floor();
	}

	setSize(width, height) {
		this._width = width;
		this._height = height;
		this._viewport.set(0, 0, width, height);
		this._scissor.set(0, 0, width, height);
		N.setSize(this._id, width, height);
	}

	setClearColor(color, alpha) {
		this._clearColor = new Color(color);
		N.setClearColor(this._id, _color.set(color).getHex(), alpha ?? 1);
	}

	getClearColor(target) {
		return target.copy(this._clearColor ?? _color.set(0x000000));
	}

	// The viewport and the scissor box (CSS pixels from the bottom left: three's), their test,
	// and autoClear, sent before each render() and clear() that changes them.
	setViewport(x, y, width, height) {
		if (x.isVector4) this._viewport.copy(x);
		else this._viewport.set(x, y, width, height);
	}

	getViewport(target) {
		return target.copy(this._viewport);
	}

	setScissor(x, y, width, height) {
		if (x.isVector4) this._scissor.copy(x);
		else this._scissor.set(x, y, width, height);
	}

	getScissor(target) {
		return target.copy(this._scissor);
	}

	setScissorTest(on) {
		this._scissorTest = on;
	}

	getScissorTest() {
		return this._scissorTest;
	}

	_viewState() {
		const v = this._viewport;
		const s = this._scissor;
		// (the whole canvas is no viewport of its own: it follows the window)
		const whole = v.x === 0 && v.y === 0 && v.z === this._width && v.w === this._height;
		const key = `${this.autoClear} ${v.x} ${v.y} ${v.z} ${v.w} ${s.x} ${s.y} ${s.z} ${s.w} ${this._scissorTest}`;
		if (this._viewKey === key) return;
		this._viewKey = key;
		N.viewState(this._id, this.autoClear, whole ? 0 : v.x, whole ? 0 : v.y, whole ? 0 : v.z, whole ? 0 : v.w,
			s.x, s.y, s.z, s.w, this._scissorTest);
	}

	// (color, depth and stencil together: drie clears them as one)
	clear() {
		this._viewState();
		N.clear(this._id);
	}

	clearColor() {
		this.clear();
	}

	clearDepth() {
		this.clear();
	}

	setAnimationLoop(callback) {
		this._loop = callback;
		// (an async callback's errors are its promise's: reported as a plain one's are)
		const loop = callback ? (t) => {
			const r = callback(t);
			if (r && typeof r.then === "function") r.then(undefined, (e) => console.error("uncaught error in the animation loop:", e));
		} : null;
		N.setAnimationLoop(this._id, loop ?? (this._last ? () => this._redraw() : null));
	}

	// A page that renders only when something changes (OrbitControls' "change" event, no
	// animation loop): its last picture is drawn again every frame, so the window has one.
	_redraw() {
		if (this._probes?.size) this._runProbes();
		if (this._last) this.render(...this._last);
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

	// three/webgpu's WebGPURenderer is this one too (js/drie_nodes.js draws its node
	// materials): nothing to wait for
	init() {
		return Promise.resolve(this);
	}

	hasInitialized() {
		return true;
	}

	compileAsync() {
		return Promise.resolve();
	}

	renderAsync(scene, camera) {
		this.render(scene, camera);
		return Promise.resolve();
	}

	render(scene, camera) {
		if (this._probes?.size) this._runProbes();
		if (!this._last && !this._loop) N.setAnimationLoop(this._id, () => this._redraw());
		this._last = [scene, camera];
		const sceneId = this._mirror(scene, camera);
		this._viewState();
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
		return (scene.__drieState ??= { mirrored: [], visited: 0, lights: [], targets: [], grids: [] });
	}

	// Brings the scene's drie mirror up to date (with `camera` in it, if given) and returns its
	// drie handle.
	// three's CubeCamera.update: the scene drawn in every direction from the camera into drie's
	// environment map (Renderer.render_probe, on the GPU), which its render target's texture
	// stands for (as an envMap, scene.environment or scene.background). Its light on surfaces
	// is worked out at the first update and every 30th (that waits for the GPU). Before the
	// first frame there is nothing to draw with: the update waits for the next render (or
	// LightProbeGenerator).
	_probe(cubeCamera, scene, at = null) {
		const texture = cubeCamera.renderTarget.texture;
		const sceneId = this._mirror(scene);
		if (cubeCamera.parent === null) cubeCamera.updateMatrixWorld();
		const p = at ?? new Vector3().setFromMatrixPosition(cubeCamera.matrixWorld);
		const face = cubeCamera.children[0];
		const count = texture.__drieProbes ?? 0;
		const id = N.renderProbe(this._id, sceneId, p.x, p.y, p.z, texture.__drieEnvironment ?? -1, cubeCamera.renderTarget.width,
			face?.near ?? 0.1, face?.far ?? 1000, count % 30 === 0);
		if (id < 0) {
			// (drawn later as the scene is now: where the camera is and what is hidden, as a
			// page hides what it doesn't want in the map just around the update)
			const hidden = [];
			scene.traverse((o) => { if (!o.visible) hidden.push(o); });
			(this._probes ??= new Map()).set(cubeCamera, { scene, at: p, hidden });
			return;
		}
		this._probes?.delete(cubeCamera);
		texture.__drieEnvironment = id;
		texture.__drieProbes = count + 1;
	}

	_runProbes() {
		for (const [cubeCamera, { scene, at, hidden }] of [...this._probes]) {
			const shown = hidden.filter((o) => o.visible);
			for (const o of shown) o.visible = false;
			try {
				this._probe(cubeCamera, scene, at);
			} finally {
				for (const o of shown) o.visible = true;
			}
		}
	}

	// LightProbeGenerator.fromCubeRenderTarget: the light on surfaces of a CubeCamera's map,
	// once it has been drawn, as three's LightProbe. (Frames run meanwhile, for a page that
	// renders once it has the probe: the first draws it.)
	async _lightProbe(cubeRenderTarget) {
		const texture = cubeRenderTarget.texture;
		const sh = new Float32Array(27);
		if (!this._last && !this._loop) N.setAnimationLoop(this._id, () => this._redraw());
		for (;;) {
			if (this._probes?.size) this._runProbes();
			const id = texture.__drieEnvironment;
			if (id !== undefined && id >= 0 && N.textureLightProbe(id, sh)) break;
			await new Promise((resolve) => setTimeout(resolve, 16));
		}
		const probe = new LightProbe();
		probe.sh.fromArray(sh);
		return probe;
	}

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
		if (globalThis.__drieNodes) {
			// (for node materials' uniforms: three's NodeFrame)
			const now = performance.now() / 1000;
			const f = (this._nodeFrame ??= { id: 0, time: 0, deltaTime: 0, start: now, last: now });
			f.id = frame;
			f.time = now - f.start;
			f.deltaTime = now - f.last;
			f.last = now;
			f.camera = camera;
			f.scene = scene;
		}
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
		let env = environment ? (environment.__drieEnvironment ?? this._texture(environment)) : -1;
		// (three/webgpu's scene.environmentNode: its graph drawn into a map, js/drie_nodes.js)
		if (scene.environmentNode && globalThis.__drieNodes) env = globalThis.__drieNodes.environment(this, scene);
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
		state.grids.length = 0;
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
		// the light probe grids that light the scene (js/light_probe_grid.js), the one with the
		// highest priority in front, as three's
		const grids = state.grids.sort((a, b) => a.priority - b.priority || a.id - b.id).slice(-4);
		const volumes = [0, 1, 2, 3].map((i) => (grids[i] ? grids[i].__drieVolume : -1));
		const key = volumes.join();
		if (scene.__drieIrradiance !== key) N.sceneIrradiance(sceneId, ...volumes), (scene.__drieIrradiance = key);
		for (let i = 0; i < state.lights.length; i++) this._light(state.lights[i]);

		// what was mirrored before and is gone from the scene now (only looked for when fewer
		// objects were seen this frame than have been mirrored)
		const mirrored = state.mirrored;
		if (state.visited !== mirrored.length) {
			for (let i = 0; i < mirrored.length; i++) {
				const o = mirrored[i];
				if (o.__drieFrame !== frame) {
					this._record(o.isSky ? o.__drieSkies.get(sceneId) : o.__drie, false, o.matrixWorld.elements);
					if (o.__drieOutline !== undefined) this._record(o.__drieOutline, false, o.matrixWorld.elements);
				}
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
			let id = o.isSky ? o.__drieSkies?.get(sceneId) : o.__drie;
			if (id === undefined) id = this._create(o, sceneId);
			o.__drieFrame = frame;
			this._scene.visited++;
			this._record(id, drawn, o.matrixWorld.elements);
			if (o.isSky) this._sky(o, id);
			else if (o.isMesh || o.isLine) this._refresh(o, frame, drawn);
			if (o.isInstancedMesh) this._instances(o);
			if (o.isLightProbeGrid && shown && o.__drieVolume !== undefined) this._scene.grids.push(o);
			if (o.isSkinnedMesh && drawn) this._skin(o, frame);
			if (o.isMesh && (this._outline || o.__drieOutline !== undefined)) this._outlineOf(o, drawn, sceneId, frame);
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
			// three's Sky shader, as drie's own Sky (its ShaderMaterial can't be drawn as it is).
			// One in each scene it is rendered in: PMREMGenerator.fromScene captures a scene
			// with the Sky (at the next render) that the application then moves into another.
			id = N.sky();
			(o.__drieSkies ??= new Map()).set(sceneId, id);
		} else if (o.isMesh || o.isLine) {
			id = this._newMesh(o, this._geometry(o.geometry, this._frame), this._material(o.material, this._frame));
			o.__drieGeometry = o.geometry.__drie;
			o.__drieMaterial = (Array.isArray(o.material) ? o.material[0] : o.material).__drie;
		} else if (o.isLightProbeGrid) {
			id = N.group();
		} else if (o.isLight) {
			_color.copy(o.color);
			// (a RectAreaLight's width and height in place of distance and decay)
			id = N.light(lightKind(o), _color.getHex(), o.intensity, o.isRectAreaLight ? o.width : o.distance ?? 0,
				o.isRectAreaLight ? o.height : o.decay ?? 2, o.angle ?? Math.PI / 3,
				o.penumbra ?? 0, o.groundColor ? o.groundColor.getHex() : 0xffffff);
			if (o.target) {
				const target = N.lightTarget(id);
				if (target !== null) this._scene.targets.push([o, target]);
			}
			this._scene.lights.push(o);
		} else if (o.isPerspectiveCamera) {
			id = N.camera(o.fov, o.aspect, o.near, o.far);
		} else if (o.isOrthographicCamera) {
			id = N.orthoCamera(o.left, o.right, o.top, o.bottom, o.near, o.far);
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

	// Sky's uniforms, in the order of N.skyParams: the material's (Sky) or its own uniform nodes
	// (SkyMesh, whose clouds move with TSL's time). (r177's Sky has no clouds: none when its
	// uniforms are missing.)
	_skyArgs(sky) {
		const u = sky.isSkyMesh ? sky : sky.material.uniforms;
		const v = (name, fallback) => (u[name] ? u[name].value : fallback);
		const sun = u.sunPosition.value;
		return [v("turbidity", 2), v("rayleigh", 1), v("mieCoefficient", 0.005), v("mieDirectionalG", 0.8),
			sun.x, sun.y, sun.z, v("cloudScale", 0.0002), v("cloudSpeed", 0.00002), v("cloudCoverage", 0),
			v("cloudDensity", 0.4), v("cloudElevation", 0.5), v("showSunDisc", 1), sky.isSkyMesh ? performance.now() / 1000 : v("time", 0)];
	}

	// Sky's uniforms, sent to its drie sky `id` when one changed.
	_sky(sky, id) {
		const args = this._skyArgs(sky);
		const key = args.join(",");
		const sent = (sky.__drieSkyKeys ??= new Map());
		if (sent.get(id) === key) return;
		sent.set(id, key);
		N.skyParams(id, ...args);
	}

	_instances(mesh) {
		const version = mesh.instanceMatrix.version;
		const colors = mesh.instanceColor;
		const colorVersion = colors ? colors.version : -1;
		if (mesh.__drieInstances === version && mesh.__drieCount === mesh.count && mesh.__drieColors === colorVersion) return;
		if (mesh.__drieInstances !== version || mesh.__drieCount !== mesh.count) {
			N.instances(mesh.__drie, mesh.instanceMatrix.array, mesh.count);
		}
		// (instanceColor: the material's color times it, as three's)
		if (colors && (mesh.__drieColors !== colorVersion || mesh.__drieCount !== mesh.count)) {
			N.instanceColors(mesh.__drie, colors.array, Math.min(mesh.count, colors.count));
		}
		mesh.__drieInstances = version;
		mesh.__drieCount = mesh.count;
		mesh.__drieColors = colorVersion;
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

	// The outline OutlineEffect draws round a mesh (js/outline_effect.js): a drie mesh of its
	// geometry, drawn with the back faces of a material of the outline's color pushed out by
	// drie's Material.outline, and only in the effect's renders. One outline material per
	// material, as three's.
	_outlineOf(mesh, drawn, sceneId, frame) {
		const effect = this._outline;
		const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
		const p = material.userData.outlineParameters ?? {};
		const shown = drawn && effect !== null && p.visible !== false && material.visible !== false
			&& mesh.geometry.attributes.normal !== undefined;
		let outline = material.__drieOutlineMaterial;
		if (shown) {
			const alpha = p.alpha ?? effect.defaultAlpha;
			if (!outline) outline = material.__drieOutlineMaterial = new MeshBasicMaterial({ side: BackSide });
			if (p.color) outline.color.fromArray(p.color);
			else outline.color.copy(effect.defaultColor);
			outline.opacity = alpha;
			if (outline.transparent !== alpha < 1 || outline.fog !== material.fog) {
				outline.transparent = alpha < 1;
				outline.fog = material.fog;
				outline.needsUpdate = true;
			}
			const g = this._geometry(mesh.geometry, frame);
			const m = this._material(outline, frame);
			const thickness = p.thickness ?? effect.defaultThickness;
			if (outline.__drieThickness !== thickness || outline.__drieOutlined !== m) {
				outline.__drieThickness = thickness;
				outline.__drieOutlined = m;
				N.materialOutline(m, thickness);
			}
			if (mesh.__drieOutline === undefined || mesh.__drieOutlineGeometry !== g || mesh.__drieOutlineMaterial !== m) {
				if (mesh.__drieOutline !== undefined) this._record(mesh.__drieOutline, false, mesh.matrixWorld.elements);
				mesh.__drieOutline = mesh.isSkinnedMesh ? N.skinnedMesh(g, m) : N.mesh(g, m);
				mesh.__drieOutlineGeometry = g;
				mesh.__drieOutlineMaterial = m;
				N.add(sceneId, mesh.__drieOutline);
				N.shadow(mesh.__drieOutline, false, false);
			}
			if (mesh.isSkinnedMesh && mesh.__drieSkin) N.skin(mesh.__drieOutline, mesh.__drieSkin, mesh.skeleton.bones.length);
		}
		if (mesh.__drieOutline !== undefined) this._record(mesh.__drieOutline, shown, mesh.matrixWorld.elements);
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
			|| (ground && (v[4] !== ground.r || v[5] !== ground.g || v[6] !== ground.b))
			|| (light.isRectAreaLight && (v[7] !== light.width || v[8] !== light.height))) {
			v[0] = light.color.r;
			v[1] = light.color.g;
			v[2] = light.color.b;
			v[3] = light.intensity;
			if (ground) {
				v[4] = ground.r;
				v[5] = ground.g;
				v[6] = ground.b;
			}
			v[7] = light.width;
			v[8] = light.height;
			N.lightUpdate(light.__drie, light.color.getHex(), light.intensity, ground ? ground.getHex() : 0,
				light.width ?? 0, light.height ?? 0);
		}
		const channels = light.__drieChannels ?? 0;
		if (light.__drieSentChannels !== channels) N.lightChannels(light.__drie, (light.__drieSentChannels = channels));
		if (!light.isDirectionalLight && !light.isSpotLight && !light.isSunLight && !light.isPointLight) return;
		const shadow = light.shadow;
		const cam = shadow.camera;
		const cast = this.shadowMap.enabled && light.castShadow;
		// (a spot light's shadow covers its cone, a point light's all round it: no box)
		const extent = light.isDirectionalLight ? Math.max(Math.abs(cam.right - cam.left), Math.abs(cam.top - cam.bottom)) / 2 : 0;
		const focus = shadow.focus ?? 1;
		// SunLight's cascades (SunLightShadow's practical splits and fade: drie's split_lambda
		// 0.5 and cascade_blend 0.1), up to its camera's far
		const cascades = light.isSunLight ? shadow._cascadeSplits.length - 1 : 1;
		const key = cast ? `${shadow.mapSize.x} ${shadow.radius} ${cam.near} ${cam.far} ${extent} ${shadow.normalBias} ${shadow.intensity} ${shadow.bias} ${shadow.blurSamples} ${focus} ${cascades}` : "";
		if (light.__drieShadowKey === key) return;
		light.__drieShadowKey = key;
		N.lightShadow(light.__drie, cast, shadow.mapSize.x, shadow.radius, cam.near, cam.far, extent, shadow.normalBias,
			shadow.intensity ?? 1, shadow.bias, shadow.blurSamples ?? 8, focus, cascades, 0.5, 0.1);
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

	// scene.background as an equirectangular texture or a cube map, or a flat picture over the
	// view (a texture of another mapping) (a color: see render()).
	_background(scene, sceneId) {
		// (three/webgpu's scene.backgroundNode: a node material drawn behind everything,
		// js/drie_nodes.js)
		const material = scene.backgroundNode && globalThis.__drieNodes ? globalThis.__drieNodes.background(this, scene) : -1;
		if (scene.__drieBackgroundMaterial !== material) {
			scene.__drieBackgroundMaterial = material;
			N.backgroundMaterial(sceneId, material);
		}
		const background = scene.background;
		// (a PMREMGenerator's texture is drie's capture of a scene)
		const texture = background && background.isTexture ? (background.__drieEnvironment ?? this._texture(background)) : -1;
		const flat = texture >= 0 && background.__drieEnvironment === undefined && !background.isCubeTexture
			&& background.mapping !== EquirectangularReflectionMapping && background.mapping !== EquirectangularRefractionMapping;
		const key = texture < 0 ? -1 : `${texture} ${scene.backgroundBlurriness} ${scene.backgroundIntensity} ${flat}`;
		if (scene.__drieBackgroundTexture === key) return;
		scene.__drieBackgroundTexture = key;
		N.backgroundTexture(sceneId, texture, scene.backgroundBlurriness ?? 0, scene.backgroundIntensity ?? 1, flat);
	}

	// An image drie loads from its file (`path`) or decodes from memory (`data`, from a Blob,
	// as GLTFLoader makes them), or a DataTexture of half floats or floats (an HDR image, as
	// UltraHDRLoader and HDRLoader make them). `equirectangular`: it is one whatever its
	// mapping says (PMREMGenerator.fromEquirectangular's).
	_texture(texture, equirectangular = false) {
		// (a texture that drie made, as the light map of js/progressive_lightmap.js)
		if (texture?.__drieNative !== undefined) return texture.__drieNative;
		if (texture?.isCubeTexture) return this._cubeTexture(texture);
		const image = texture?.image;
		// (a canvas: the pixels its 2D context drew; a DataTexture of bytes: them, as RGBA)
		let canvas = image?.tagName === "CANVAS" ? image.getContext("2d")._buffer() : null;
		if (!canvas && texture?.isDataTexture && (image?.data instanceof Uint8Array || image?.data instanceof Uint8ClampedArray)) {
			const data = rgbaBytes(image, texture.format);
			if (data) canvas = { data, width: image.width, height: image.height };
		}
		const source = canvas ? canvas.data : image ? (image.data ?? image.path) : undefined;
		if (source === undefined) return -1;
		if (texture.__drie !== undefined && texture.__drieVersion === texture.version) {
			// (repeat, offset, rotation and center change without a new version)
			const uv = uvKey(texture);
			if (texture.__drieUv !== uv) {
				const { repeat, offset, center } = texture;
				N.textureUv(texture.__drie, repeat.x, repeat.y, offset.x, offset.y, texture.rotation, center.x, center.y);
				texture.__drieUv = uv;
			}
			return texture.__drie;
		}
		// (three's equirectangular maps have the sky at v = 1, drie's at v = 0: flipped the other way)
		const equirect = equirectangular || texture.mapping === EquirectangularReflectionMapping
			|| texture.mapping === EquirectangularRefractionMapping;
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
					canvas ? canvas.height : 0, texture.magFilter === NearestFilter, texture.channel ?? 0,
					texture.minFilter, texture.generateMipmaps);
				if (canvas) this._mipmaps(texture);
				images?.set(key, texture.__drie);
			}
		}
		texture.__drieVersion = texture.version;
		texture.__drieUv = uvKey(texture);
		return texture.__drie;
	}

	// The mip levels the application drew (texture.mipmaps after the first), canvases or
	// bytes, for a texture of pixels.
	_mipmaps(texture) {
		const levels = texture.mipmaps ?? [];
		for (let i = 1; i < levels.length; i++) {
			const level = levels[i];
			const pixels = level?.tagName === "CANVAS" ? level.getContext("2d")._buffer() : level;
			if (!(pixels?.data instanceof Uint8Array || pixels?.data instanceof Uint8ClampedArray)) break;
			N.textureMipmap(texture.__drie, pixels.data, pixels.width, pixels.height);
		}
	}

	// A map on the first or the second uv set (three's texture.channel; drie's geometries have
	// only those two).
	_map(texture) {
		return texture && texture.channel > 1 ? -1 : this._uvTexture(texture);
	}

	// A texture read by the geometry's uv. One whose mapping is equirectangular is drie's
	// environment map, upside down as a picture (see _texture): read by uv (GroundedSkybox's
	// map) it is a clone of it turned back by its uv transform.
	_uvTexture(texture) {
		const id = this._texture(texture);
		if (id < 0 || (texture.mapping !== EquirectangularReflectionMapping && texture.mapping !== EquirectangularRefractionMapping)) {
			return id;
		}
		const key = `${id} ${uvKey(texture)}`;
		if (texture.__drieUvKey === key) return texture.__drieUvTexture;
		const { repeat, offset, center } = texture;
		texture.__drieUvTexture = N.textureClone(id, repeat.x, -repeat.y, offset.x, 1 - 2 * center.y - offset.y,
			texture.rotation, center.x, center.y, texture.channel ?? 0);
		texture.__drieUvKey = key;
		return texture.__drieUvTexture;
	}

	// A CubeTexture, as drie's equirectangular map of it: six images (CubeTextureLoader's) or
	// six DataTextures of half floats or floats (HDRCubeTextureLoader's).
	_cubeTexture(texture) {
		// (a CubeCamera's, drawn by drie: see _probe)
		if (texture.isRenderTargetTexture) return -1;
		if (texture.__drie !== undefined && texture.__drieVersion === texture.version) return texture.__drie;
		const faces = texture.images ?? texture.image;
		if (!Array.isArray(faces) || faces.length !== 6 || faces.some((f) => !f)) return -1;
		if (faces.every((f) => f.tagName === "IMG")) {
			if (!faces.every((f) => f.complete)) return -1;
			texture.__drie = N.cubeImages(...faces.map((f) => f.data ?? f.path), texture.colorSpace === "srgb");
		} else {
			const data = faces.map((f) => f?.image?.data ?? f?.data);
			const size = faces[0]?.image?.width ?? faces[0]?.width;
			if (!size || data.some((d) => !(d instanceof Uint16Array || d instanceof Float32Array))) {
				throw new Error("driejs: a cube map's faces are images, or DataTextures of half floats or floats");
			}
			texture.__drie = N.cubeTexture(...data, size);
		}
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
			// (its own surroundings, instead of the scene's environment: its envMap, or the map
			// its envNode reads)
			material.envMap ? (material.envMap.__drieEnvironment ?? this._texture(material.envMap))
				: material.envNode ? globalThis.__drieNodes.envNodeTexture(this, material.envNode) : -1,
			material.isMeshStandardMaterial ? (material.envMapIntensity ?? 1) : 1];
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

	// How a basic, lambert or phong material's envMap shows (uniforms in three), sent when it
	// changes.
	_env(material) {
		if (material.isMeshStandardMaterial || !material.envMap) return;
		const refraction = material.envMap.mapping === CubeRefractionMapping
			|| material.envMap.mapping === EquirectangularRefractionMapping;
		const specular = this._map(material.specularMap);
		const key = `${material.reflectivity} ${material.combine} ${refraction} ${material.refractionRatio} ${specular}`;
		if (material.__drieEnv === key) return;
		material.__drieEnv = key;
		N.materialEnv(material.__drie, material.reflectivity ?? 1, material.combine ?? 0, refraction,
			material.refractionRatio ?? 0.98, specular);
	}

	// bumpMap and bumpScale, matcap and gradientMap, sent when they change.
	_surface(material) {
		const bump = this._map(material.bumpMap);
		const matcap = this._texture(material.matcap);
		const gradient = this._texture(material.gradientMap);
		const displacement = this._map(material.displacementMap);
		const key = `${bump} ${material.bumpScale} ${matcap} ${gradient} ${displacement} ${material.displacementScale} ${material.displacementBias} ${material.depthPacking}`;
		if (material.__drieSurface === key) return;
		material.__drieSurface = key;
		N.materialSurface(material.__drie, bump, material.bumpScale ?? 1, matcap, gradient, displacement,
			material.displacementScale ?? 1, material.displacementBias ?? 0, material.depthPacking ?? 3200);
	}

	// blending, premultipliedAlpha and the custom blending's equations and factors (three sets
	// them for each draw), sent when they change.
	_blend(material) {
		const m = material;
		const key = `${m.blending} ${m.premultipliedAlpha} ${m.blendEquation} ${m.blendSrc} ${m.blendDst} ${m.blendEquationAlpha} ${m.blendSrcAlpha} ${m.blendDstAlpha}`;
		if (m.__drieBlend === key) return;
		m.__drieBlend = key;
		N.materialBlend(m.__drie, m.blending ?? 1, m.premultipliedAlpha === true, m.blendEquation ?? 100, m.blendSrc ?? 204,
			m.blendDst ?? 205, m.blendEquationAlpha ?? m.blendEquation ?? 100, m.blendSrcAlpha ?? m.blendSrc ?? 204,
			m.blendDstAlpha ?? m.blendDst ?? 205);
	}

	// depthTest, depthWrite and toneMapped (three sets them for each draw: no needsUpdate either).
	_depth(material) {
		const key = (material.depthTest === false ? 1 : 0) + (material.depthWrite === false ? 2 : 0)
			+ (material.toneMapped === false ? 4 : 0);
		if (material.__drieDepth === key) return;
		material.__drieDepth = key;
		N.materialDepth(material.__drie, material.depthTest !== false, material.depthWrite !== false,
			material.toneMapped !== false);
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

	// A node material's lightsNode (three's lights([...])): the lights it lists, and only them,
	// light it. Each list is a lighting channel of drie's (Light.channels,
	// Material.light_channels), up to 24.
	_lightsNode(material) {
		const lights = material.lightsNode?.getLights?.();
		let channels = 0;
		if (lights) {
			const groups = (this._lightGroups ??= new Map());
			const key = lights.map((l) => l.id).join();
			let bit = groups.get(key);
			if (bit === undefined) {
				bit = groups.size;
				if (bit >= 24) throw new Error("driejs: more than 24 different lists of lights (lightsNode)");
				groups.set(key, bit);
			}
			channels = 2 ** bit;
			for (const l of lights) l.__drieChannels = (l.__drieChannels ?? 0) | channels;
		}
		if (material.__drieLightChannels !== channels) {
			material.__drieLightChannels = channels;
			N.materialLightChannels(material.__drie, channels);
		}
	}

	_material(material, frame) {
		if (Array.isArray(material)) material = material[0];
		if (material.__drieFrame === frame) return material.__drie;
		material.__drieFrame = frame;
		const map = this._uvTexture(material.map);
		if (material.__drie !== undefined && material.__drieVersion === material.version && material.__drieMap === map) {
			this._materialValues(material);
			this._maps(material);
			this._layers(material);
			this._lightMap(material);
			this._lightsNode(material);
			this._env(material);
			this._surface(material);
			this._blend(material);
			this._depth(material);
			if (this._hasNodes(material)) globalThis.__drieNodes.sync(this, material, this._nodeFrame);
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
		material.__drieLightChannels = undefined;
		material.__drieDepth = undefined;
		material.__drieEnv = undefined;
		material.__drieSurface = undefined;
		material.__drieBlend = undefined;
		this._materialValues(material);
		this._maps(material);
		this._layers(material);
		this._lightMap(material);
		this._lightsNode(material);
		this._env(material);
		this._surface(material);
		this._blend(material);
		this._depth(material);
		if (this._hasNodes(material)) globalThis.__drieNodes.sync(this, material, this._nodeFrame);
		return material.__drie;
	}

	// Whether drie draws the material with TSL graphs (js/drie_nodes.js): three/webgpu's node
	// materials, and any material in a scene with a fogNode (or that was in one)
	_hasNodes(material) {
		if (!globalThis.__drieNodes) return false;
		return material.isNodeMaterial || !!this._nodeFrame?.scene?.fogNode || material.__drieNodes !== undefined;
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
		const c = camera;
		const key = `${c.fov} ${c.aspect} ${c.near} ${c.far} ${c.zoom} ${c.left} ${c.right} ${c.top} ${c.bottom}`;
		if (c.__drieProjection === key) return;
		c.__drieProjection = key;
		N.cameraUpdate(c.__drie, c.fov ?? 50, c.aspect ?? 1, c.near, c.far, c.zoom, c.left ?? -1, c.right ?? 1, c.top ?? 1,
			c.bottom ?? -1);
	}
}

// (see DrieRenderer._probe)
CubeCamera.prototype.update = function (renderer, scene) {
	renderer._probe(this, scene);
};
