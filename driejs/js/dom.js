// What three.js and its examples expect of a browser, without one: a window of the
// renderer's default size, a document of elements that take listeners and children, the
// window's mouse and keyboard as DOM events (bubbling up from the canvas, or from the
// element holding the pointer lock), images and files that drie and the host read, and
// fetch, TextDecoder, Blob and object URLs as GLTFLoader uses them, and DOMParser for XML.
// Imported by the "three" module before three.js itself.
const N = globalThis.__drie;

// ----------------------------------------------------------------------------
// Events and elements
// ----------------------------------------------------------------------------

class Target {
	addEventListener(type, fn) {
		if (!fn) return;
		const listeners = (this._listeners ??= {});
		const list = (listeners[type] ??= []);
		if (!list.includes(fn)) list.push(fn);
	}
	removeEventListener(type, fn) {
		const list = this._listeners?.[type];
		const i = list ? list.indexOf(fn) : -1;
		if (i >= 0) list.splice(i, 1);
	}
	// Calls this target's listeners only; see dispatch() for bubbling.
	dispatchEvent(event) {
		const list = this._listeners?.[event.type];
		if (!list) return true;
		event.currentTarget = this;
		for (const fn of list.slice()) {
			if (typeof fn === "function") fn.call(this, event);
			else fn.handleEvent(event);
		}
		return !event.defaultPrevented;
	}
}

class Element extends Target {
	constructor(tagName) {
		super();
		this.tagName = tagName.toUpperCase();
		this.nodeName = this.tagName;
		this.style = {};
		this.children = [];
		this.childNodes = this.children;
		this.parentNode = null;
		this.width = 0;
		this.height = 0;
		this.classList = { add() {}, remove() {}, toggle() {}, contains: () => false };
		this.dataset = {};
	}
	get parentElement() {
		return this.parentNode instanceof Element ? this.parentNode : null;
	}
	get clientWidth() {
		return innerWidth;
	}
	get clientHeight() {
		return innerHeight;
	}
	getBoundingClientRect() {
		return { left: 0, top: 0, x: 0, y: 0, right: innerWidth, bottom: innerHeight, width: innerWidth, height: innerHeight };
	}
	appendChild(child) {
		if (child.parentNode) child.parentNode.removeChild(child);
		this.children.push(child);
		child.parentNode = this;
		return child;
	}
	insertBefore(child) {
		return this.appendChild(child);
	}
	removeChild(child) {
		const i = this.children.indexOf(child);
		if (i >= 0) this.children.splice(i, 1);
		child.parentNode = null;
		return child;
	}
	remove() {
		this.parentNode?.removeChild(this);
	}
	setAttribute() {}
	getAttribute() {
		return null;
	}
	focus() {}
	blur() {}
	setPointerCapture() {}
	releasePointerCapture() {}
	hasPointerCapture() {
		return false;
	}
	requestPointerLock() {
		if (document.pointerLockElement === this) return;
		document.pointerLockElement = this;
		N.lockMouse(true);
		document.dispatchEvent({ type: "pointerlockchange", target: document });
	}
	getRootNode() {
		return document;
	}
	get ownerDocument() {
		return document;
	}
	// A canvas's 2D context (see Context2D): the one thing it draws is images.
	getContext(kind) {
		if (this.tagName !== "CANVAS" || kind !== "2d") return null;
		return (this._context2d ??= new Context2D(this));
	}
}

// What createImageBitmap makes: the encoded image (`data`, which the renderer hands drie as an
// Image's), decoded by drie only when its size or pixels are asked for (a canvas drawing it).
class ImageBitmap {
	constructor(bytes) {
		this.data = bytes;
		this._decoded = undefined;
	}
	_handle() {
		if (this._decoded === undefined) {
			this._decoded = N.imageDecode(this.data);
			if (this._decoded < 0) throw new Error("createImageBitmap: the image cannot be decoded");
		}
		return this._decoded;
	}
	get width() {
		return N.imageSize(this._handle(), 0);
	}
	get height() {
		return N.imageSize(this._handle(), 1);
	}
	close() {
		if (this._decoded !== undefined && this._decoded >= 0) N.imageFree(this._decoded);
		this._decoded = -1;
	}
}

// A canvas's 2D context as UltraHDRLoader uses it: drawImage of an ImageBitmap (stretched to
// any size, bilinear) and getImageData. Its pixels are made when the canvas's size is first
// drawn at.
class Context2D {
	constructor(canvas) {
		this.canvas = canvas;
		this._pixels = null;
	}
	_buffer() {
		const { width, height } = this.canvas;
		if (!this._pixels || this._pixels.width !== width || this._pixels.height !== height) {
			this._pixels = { width, height, data: new Uint8ClampedArray(width * height * 4) };
		}
		return this._pixels;
	}
	drawImage(image, ...args) {
		let sx = 0, sy = 0, sw = image.width, sh = image.height, dx, dy, dw = sw, dh = sh;
		if (args.length === 2) [dx, dy] = args;
		else if (args.length === 4) [dx, dy, dw, dh] = args;
		else [sx, sy, sw, sh, dx, dy, dw, dh] = args;
		if (sx !== 0 || sy !== 0 || sw !== image.width || sh !== image.height) {
			throw new Error("driejs: drawImage of part of an image is not supported");
		}
		const target = this._buffer();
		dx = Math.round(dx);
		dy = Math.round(dy);
		dw = Math.round(dw);
		dh = Math.round(dh);
		if (dx === 0 && dy === 0 && dw === target.width && dh === target.height) {
			N.imagePixels(image._handle(), target.data, dw, dh);
			return;
		}
		const pixels = new Uint8ClampedArray(dw * dh * 4);
		N.imagePixels(image._handle(), pixels, dw, dh);
		for (let y = Math.max(0, -dy); y < dh && y + dy < target.height; y++) {
			const from = y * dw * 4;
			const x0 = Math.max(0, -dx);
			const x1 = Math.min(dw, target.width - dx);
			if (x1 > x0) target.data.set(pixels.subarray(from + x0 * 4, from + x1 * 4), ((y + dy) * target.width + dx + x0) * 4);
		}
	}
	getImageData(x, y, width, height) {
		const target = this._buffer();
		if (x === 0 && y === 0 && width === target.width && height === target.height) {
			return { width, height, data: target.data.slice(), colorSpace: "srgb" };
		}
		const data = new Uint8ClampedArray(width * height * 4);
		for (let row = 0; row < height; row++) {
			const from = ((row + y) * target.width + x) * 4;
			data.set(target.data.subarray(from, from + width * 4), row * width * 4);
		}
		return { width, height, data, colorSpace: "srgb" };
	}
}

// An image is not decoded here: the renderer hands drie its `path`, or the bytes of a Blob
// it came from (`data`). `load` fires once the current job is done, as in a browser.
class Image extends Element {
	constructor() {
		super("img");
		this._src = "";
		this.complete = false;
	}
	get src() {
		return this._src;
	}
	set src(url) {
		this._src = url;
		const blob = blobs.get(url);
		if (blob) this.data = blob.bytes();
		else this.path = url.startsWith("/") ? url : N.baseDir + url;
		Promise.resolve().then(() => {
			this.complete = true;
			this.dispatchEvent({ type: "load", target: this });
		});
	}
}

const create = (tagName) => (tagName.toLowerCase() === "img" ? new Image() : new Element(tagName));

class Document extends Target {
	constructor() {
		super();
		this.title = "three.js";
		this.documentElement = new Element("html");
		this.documentElement.parentNode = this;
		this.body = this.documentElement.appendChild(new Element("body"));
		this.pointerLockElement = null;
		this._ids = {};
	}
	createElement(tagName) {
		return create(tagName);
	}
	createElementNS(ns, tagName) {
		return create(tagName);
	}
	// The pages' elements by id: each made, inside the body, the first time it is asked for.
	getElementById(id) {
		let element = this._ids[id];
		if (!element) {
			element = this._ids[id] = this.body.appendChild(new Element("div"));
			element.id = id;
		}
		return element;
	}
	querySelector(selector) {
		return selector.startsWith("#") ? this.getElementById(selector.slice(1)) : null;
	}
	exitPointerLock() {
		if (this.pointerLockElement === null) return;
		this.pointerLockElement = null;
		N.lockMouse(false);
		this.dispatchEvent({ type: "pointerlockchange", target: this });
	}
}

globalThis.document = new Document();
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 720;
globalThis.devicePixelRatio = 1;
const windowTarget = new Target();
globalThis.addEventListener = (type, fn) => windowTarget.addEventListener(type, fn);
globalThis.removeEventListener = (type, fn) => windowTarget.removeEventListener(type, fn);
globalThis.navigator = { userAgent: "boomkat" };
globalThis.performance = { now: () => N.now() };
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};

// The renderer's canvas: where mouse events start when the pointer is not locked.
let canvas = null;
export function setCanvas(element) {
	canvas = element;
}

// From `target` up through its parents to the document, then the window.
function dispatch(event, target) {
	event.target = target;
	event.defaultPrevented = false;
	event.preventDefault = () => {
		event.defaultPrevented = true;
	};
	event.stopPropagation = () => {
		event.stopped = true;
	};
	for (let t = target; t; t = t.parentNode) {
		t.dispatchEvent(event);
		if (event.stopped) return;
	}
	if (target !== windowTarget) windowTarget.dispatchEvent(event);
}

// ----------------------------------------------------------------------------
// Mouse and keyboard, from the host
// ----------------------------------------------------------------------------

let buttons = 0;
const modifiers = { ctrlKey: false, shiftKey: false, altKey: false, metaKey: false };

function mouseEvent(type, x, y, button, dx, dy) {
	return {
		type, clientX: x, clientY: y, pageX: x, pageY: y, offsetX: x, offsetY: y, screenX: x, screenY: y,
		movementX: dx, movementY: dy, button, buttons, pointerId: 1, pointerType: "mouse", isPrimary: true,
		...modifiers,
	};
}

const BUTTON_BITS = [1, 2, 4];

// sokol's key codes (GLFW's) as KeyboardEvent.code and .key
function keyCode(k) {
	if (k >= 65 && k <= 90) return [`Key${String.fromCharCode(k)}`, String.fromCharCode(k + 32)];
	if (k >= 48 && k <= 57) return [`Digit${k - 48}`, String(k - 48)];
	if (k >= 290 && k <= 314) return [`F${k - 289}`, `F${k - 289}`];
	switch (k) {
		case 32: return ["Space", " "];
		case 256: return ["Escape", "Escape"];
		case 257: return ["Enter", "Enter"];
		case 258: return ["Tab", "Tab"];
		case 259: return ["Backspace", "Backspace"];
		case 261: return ["Delete", "Delete"];
		case 262: return ["ArrowRight", "ArrowRight"];
		case 263: return ["ArrowLeft", "ArrowLeft"];
		case 264: return ["ArrowDown", "ArrowDown"];
		case 265: return ["ArrowUp", "ArrowUp"];
		case 340: return ["ShiftLeft", "Shift"];
		case 341: return ["ControlLeft", "Control"];
		case 342: return ["AltLeft", "Alt"];
		case 343: return ["MetaLeft", "Meta"];
		case 344: return ["ShiftRight", "Shift"];
		case 345: return ["ControlRight", "Control"];
		case 346: return ["AltRight", "Alt"];
		case 347: return ["MetaRight", "Meta"];
		default: return ["Unidentified", "Unidentified"];
	}
}

const MODIFIER_OF = { Shift: "shiftKey", Control: "ctrlKey", Alt: "altKey", Meta: "metaKey" };

// Called by the host: kind 0 mouse move, 1 down, 2 up, 3 wheel (dx, dy: scroll), 4 key
// down, 5 key up (a: sokol's key code, dx: 1 for a repeat). x and y in window points.
globalThis.__drieInput = (kind, x, y, a, dx, dy) => {
	const target = document.pointerLockElement ?? canvas ?? document.body;
	switch (kind) {
		case 0:
			dispatch(mouseEvent("pointermove", x, y, -1, dx, dy), target);
			dispatch(mouseEvent("mousemove", x, y, 0, dx, dy), target);
			break;
		case 1:
			buttons |= BUTTON_BITS[a] ?? 0;
			dispatch(mouseEvent("pointerdown", x, y, a, 0, 0), target);
			dispatch(mouseEvent("mousedown", x, y, a, 0, 0), target);
			break;
		case 2:
			buttons &= ~(BUTTON_BITS[a] ?? 0);
			dispatch(mouseEvent("pointerup", x, y, a, 0, 0), target);
			dispatch(mouseEvent("mouseup", x, y, a, 0, 0), target);
			if (a === 0) dispatch(mouseEvent("click", x, y, 0, 0, 0), target);
			break;
		case 3: {
			const e = mouseEvent("wheel", x, y, 0, 0, 0);
			// a notch of a wheel is about 100 pixels in a browser, and up is negative
			e.deltaX = -dx * 100;
			e.deltaY = -dy * 100;
			e.deltaMode = 0;
			dispatch(e, target);
			break;
		}
		case 4:
		case 5: {
			const [code, key] = keyCode(a);
			const down = kind === 4;
			if (MODIFIER_OF[key]) modifiers[MODIFIER_OF[key]] = down;
			const e = { type: down ? "keydown" : "keyup", code, key, repeat: dx === 1, ...modifiers };
			dispatch(e, document.body);
			// as in a browser, Escape lets go of the pointer
			if (down && code === "Escape") document.exitPointerLock();
			break;
		}
	}
};

// ----------------------------------------------------------------------------
// Files: fetch, Blob, object URLs, TextDecoder
// ----------------------------------------------------------------------------

function readFile(url) {
	const path = url.startsWith("/") ? url : N.baseDir + url.replace(/^\.\//, "");
	const size = N.fileSize(path);
	if (size < 0) return null;
	const bytes = new Uint8Array(size);
	N.readFile(path, bytes);
	return bytes;
}

class Headers {
	constructor(init) {
		this._map = {};
		if (init) for (const k in init) this._map[k.toLowerCase()] = String(init[k]);
	}
	get(name) {
		return this._map[name.toLowerCase()] ?? null;
	}
	set(name, value) {
		this._map[name.toLowerCase()] = String(value);
	}
	has(name) {
		return name.toLowerCase() in this._map;
	}
}

class Request {
	constructor(url, init = {}) {
		this.url = typeof url === "string" ? url : url.url;
		this.headers = init.headers ?? new Headers();
	}
}

class Response {
	constructor(bytes, url) {
		this._bytes = bytes;
		this.url = url;
		this.status = bytes ? 200 : 404;
		this.ok = bytes !== null;
		this.statusText = bytes ? "OK" : "Not Found";
		this.headers = new Headers({ "Content-Length": bytes ? bytes.length : 0 });
		this.body = undefined;
	}
	async arrayBuffer() {
		const b = this._bytes;
		return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
	}
	async text() {
		return new TextDecoder().decode(this._bytes);
	}
	async json() {
		return JSON.parse(await this.text());
	}
	async blob() {
		return new Blob([this._bytes]);
	}
}

class Blob {
	constructor(parts = [], options = {}) {
		this.type = options.type ?? "";
		this._parts = parts;
		this.size = parts.reduce((n, p) => n + (typeof p === "string" ? p.length : p.byteLength), 0);
	}
	bytes() {
		const out = new Uint8Array(this.size);
		let at = 0;
		for (const p of this._parts) {
			const b = typeof p === "string" ? new TextEncoder().encode(p)
				: p instanceof ArrayBuffer ? new Uint8Array(p) : new Uint8Array(p.buffer, p.byteOffset, p.byteLength);
			out.set(b, at);
			at += b.length;
		}
		return out;
	}
	async arrayBuffer() {
		return this.bytes().buffer;
	}
}

const blobs = new Map();
let nextBlob = 1;

globalThis.URL ??= class URL {
	constructor(url, base) {
		this.href = base && !/^[a-z]+:/.test(url) ? base.replace(/[^/]*$/, "") + url : url;
	}
	toString() {
		return this.href;
	}
};
URL.createObjectURL = (blob) => {
	const url = `blob:driejs/${nextBlob++}`;
	blobs.set(url, blob);
	return url;
};
URL.revokeObjectURL = (url) => blobs.delete(url);

// The structured clone that postMessage and structuredClone make: plain objects, arrays,
// typed arrays, ArrayBuffers, Maps, Sets, Dates and primitives, with shared and cyclic
// references kept. An ArrayBuffer in `transfer` moves instead of being copied. A function
// can't be cloned (DataCloneError, as in a browser).
function cloneMessage(value, transfer = [], seen = new Map()) {
	if (value === null || typeof value !== "object") {
		if (typeof value === "function") throw new DOMException("function could not be cloned", "DataCloneError");
		return value;
	}
	if (seen.has(value)) return seen.get(value);
	let out;
	if (value instanceof ArrayBuffer) {
		out = transfer.includes(value) ? value : value.slice(0);
	} else if (ArrayBuffer.isView(value)) {
		const buffer = cloneMessage(value.buffer, transfer, seen);
		out = value instanceof DataView ? new DataView(buffer, value.byteOffset, value.byteLength)
			: new value.constructor(buffer, value.byteOffset, value.length);
	} else if (value instanceof Date) {
		out = new Date(value.getTime());
	} else if (value instanceof Map) {
		out = new Map();
		seen.set(value, out);
		for (const [k, v] of value) out.set(cloneMessage(k, transfer, seen), cloneMessage(v, transfer, seen));
		return out;
	} else if (value instanceof Set) {
		out = new Set();
		seen.set(value, out);
		for (const v of value) out.add(cloneMessage(v, transfer, seen));
		return out;
	} else {
		out = Array.isArray(value) ? [] : {};
		seen.set(value, out);
		for (const key of Object.keys(value)) out[key] = cloneMessage(value[key], transfer, seen);
		return out;
	}
	seen.set(value, out);
	return out;
}

globalThis.DOMException ??= class DOMException extends Error {
	constructor(message = "", name = "Error") {
		super(message);
		this.name = name;
	}
};
globalThis.structuredClone ??= (value, options) => cloneMessage(value, options?.transfer ?? []);

// AbortController and AbortSignal, for three's loaders (their fetches are not cancelled: each
// is a file read that has finished by the time it could be aborted).
class AbortSignal extends Target {
	constructor() {
		super();
		this.aborted = false;
		this.reason = undefined;
		this.onabort = null;
	}
	throwIfAborted() {
		if (this.aborted) throw this.reason;
	}
	static abort(reason) {
		const c = new AbortController();
		c.abort(reason);
		return c.signal;
	}
}

class AbortController {
	constructor() {
		this.signal = new AbortSignal();
	}
	abort(reason = new DOMException("This operation was aborted", "AbortError")) {
		const signal = this.signal;
		if (signal.aborted) return;
		signal.aborted = true;
		signal.reason = reason;
		const event = { type: "abort", target: signal };
		signal.onabort?.(event);
		signal.dispatchEvent(event);
	}
}
globalThis.AbortSignal ??= AbortSignal;
globalThis.AbortController ??= AbortController;

// A Worker that runs on this thread, for three's loaders (DRACOLoader's decoder): its script
// (a blob or a file) runs once, in a function of its own, where `self` is the worker's global
// (inheriting this one's) and `onmessage` is local to it. Messages go both ways on promise jobs, as they would arrive in a
// browser, structured-cloned as a browser clones them (a transferred buffer is shared rather
// than detached).
class Worker extends Target {
	constructor(url) {
		super();
		url = String(url);
		const blob = blobs.get(url);
		const source = new TextDecoder().decode(blob ? blob.bytes() : readFile(url));
		const outside = this;
		// the worker's global: everything this one has (self.Float32Array, ...), plus its
		// own events and postMessage
		const inside = Object.create(globalThis);
		for (const name of ["addEventListener", "removeEventListener", "dispatchEvent"]) inside[name] = Target.prototype[name];
		inside.postMessage = (data, transfer) => {
			const copy = cloneMessage(data, Array.isArray(transfer) ? transfer : transfer?.transfer ?? []);
			Promise.resolve().then(() => outside._receive(copy));
		};
		inside.close = () => {};
		this._inside = inside;
		if (source.includes("function decodeGeometry(") && source.includes("DracoDecoderModule(")) {
			this._handler = dracoWorker(inside);
		} else {
			const run = new Function("self", `var onmessage;\n${source}\n;return onmessage;`);
			this._handler = run.call(inside, inside) ?? inside.onmessage;
		}
		this._closed = false;
	}
	postMessage(data, transfer) {
		if (this._closed) return;
		data = cloneMessage(data, Array.isArray(transfer) ? transfer : transfer?.transfer ?? []);
		Promise.resolve().then(() => {
			const event = { type: "message", data, target: this._inside };
			if (typeof this._handler === "function") this._handler.call(this._inside, event);
			this._inside.dispatchEvent(event);
		});
	}
	_receive(data) {
		if (this._closed) return;
		const event = { type: "message", data, target: this };
		if (typeof this.onmessage === "function") this.onmessage(event);
		this.dispatchEvent(event);
	}
	terminate() {
		this._closed = true;
	}
}
globalThis.Worker = Worker;

// DRACOLoader's worker, on libdraco: the same messages and results as its DRACOWorker, which
// would otherwise run Draco's Emscripten build in JS. Its decoder script is still fetched, but
// never runs.
const DRACO_ATTRIBUTE_TYPES = { POSITION: 0, NORMAL: 1, COLOR: 2, TEX_COORD: 3, GENERIC: 4 };

function dracoWorker(self) {
	return (e) => {
		const message = e.data;
		if (message.type !== "decode") return;
		let handle = -1;
		try {
			const buffer = message.buffer;
			handle = N.dracoDecode(new Int8Array(buffer));
			const geometry = dracoGeometry(handle, message.taskConfig);
			const buffers = geometry.attributes.map((attr) => attr.array.buffer);
			if (geometry.index) buffers.push(geometry.index.array.buffer);
			self.postMessage({ type: "decode", id: message.id, geometry }, buffers);
		} catch (error) {
			console.error(error);
			self.postMessage({ type: "error", id: message.id, error: error.message });
		} finally {
			if (handle >= 0) N.dracoFree(handle);
		}
	};
}

function dracoGeometry(handle, taskConfig) {
	const { attributeIDs, attributeTypes } = taskConfig;
	const count = N.dracoInfo(handle, 1);
	const geometry = { index: null, attributes: [] };
	for (const name in attributeIDs) {
		const TypedArray = globalThis[attributeTypes[name]];
		const id = taskConfig.useUniqueIDs ? attributeIDs[name] : DRACO_ATTRIBUTE_TYPES[attributeIDs[name]];
		const attribute = N.dracoAttribute(handle, !!taskConfig.useUniqueIDs, id ?? -1);
		if (attribute < 0) {
			if (taskConfig.useUniqueIDs) throw new Error(`THREE.DRACOLoader: No attribute with unique id ${id}.`);
			continue;
		}
		const itemSize = N.dracoComponents(handle, attribute);
		const src = new TypedArray(count * itemSize);
		N.dracoAttributeData(handle, attribute, src);
		// rows padded to 4 bytes, as glTF requires (an InterleavedBufferAttribute)
		const stride = Math.ceil((itemSize * TypedArray.BYTES_PER_ELEMENT) / 4) * 4 / TypedArray.BYTES_PER_ELEMENT;
		let array = src;
		if (stride !== itemSize) {
			array = new TypedArray(count * stride);
			for (let i = 0; i < count; i++) for (let j = 0; j < itemSize; j++) array[i * stride + j] = src[i * itemSize + j];
		}
		const result = { name, count, itemSize, array, stride };
		if (name === "color") result.vertexColorSpace = taskConfig.vertexColorSpace;
		geometry.attributes.push(result);
	}
	if (N.dracoInfo(handle, 0)) {
		const index = new Uint32Array(N.dracoInfo(handle, 2) * 3);
		N.dracoIndices(handle, index);
		geometry.index = { array: index, itemSize: 1 };
	}
	return geometry;
}

globalThis.Headers = Headers;
globalThis.Request = Request;
globalThis.Response = Response;
globalThis.Blob = Blob;
globalThis.ImageBitmap = ImageBitmap;
// (GLTFLoader then loads its textures with an ImageBitmapLoader: their bytes reach drie as
// an Image's do)
globalThis.createImageBitmap = async (source) => {
	if (source instanceof ImageBitmap) return source;
	if (source instanceof Blob) return new ImageBitmap(source.bytes());
	if (source && source.data instanceof Uint8Array) return new ImageBitmap(source.data);
	throw new Error("driejs: createImageBitmap takes a Blob");
};
globalThis.fetch = (input) => {
	let url = typeof input === "string" ? input : input.url;
	// a URL made from an addon's import.meta.url (its specifier), as the host imports it
	if (url.startsWith("three/addons/")) url = N.rootDir + "vendor/jsm/" + url.slice("three/addons/".length);
	const blob = blobs.get(url);
	return Promise.resolve(new Response(blob ? blob.bytes() : readFile(url), url));
};

// UTF-8 only, which is all glTF and three's loaders use.
globalThis.TextDecoder ??= class TextDecoder {
	decode(input) {
		if (!input) return "";
		const b = input instanceof ArrayBuffer ? new Uint8Array(input) : new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
		let s = "";
		let chunk = [];
		for (let i = 0; i < b.length;) {
			let c = b[i++];
			if (c >= 0xf0) c = ((c & 7) << 18) | ((b[i++] & 63) << 12) | ((b[i++] & 63) << 6) | (b[i++] & 63);
			else if (c >= 0xe0) c = ((c & 15) << 12) | ((b[i++] & 63) << 6) | (b[i++] & 63);
			else if (c >= 0xc0) c = ((c & 31) << 6) | (b[i++] & 63);
			if (c >= 0x10000) {
				c -= 0x10000;
				chunk.push(0xd800 + (c >> 10), 0xdc00 + (c & 1023));
			} else {
				chunk.push(c);
			}
			if (chunk.length >= 4096) {
				s += String.fromCharCode(...chunk);
				chunk = [];
			}
		}
		return s + String.fromCharCode(...chunk);
	}
};

globalThis.TextEncoder ??= class TextEncoder {
	encode(s = "") {
		const out = [];
		for (const ch of s) {
			const c = ch.codePointAt(0);
			if (c < 0x80) out.push(c);
			else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
			else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
			else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
		}
		return new Uint8Array(out);
	}
};

// ----------------------------------------------------------------------------
// XML: DOMParser, for three's loaders of XML formats (Collada, 3MF, AMF, VTK, ...)
// ----------------------------------------------------------------------------

// The host parses (N.xmlParse, std::encoding::xml in src/xml.c3) and hands back the
// document's events; the nodes are made here. XML only: a "text/html" string is parsed as
// XML too. Comments are left out, CDATA sections are text, and a document that fails to
// parse has a <parsererror> element holding a <div> with the message, as in Chrome.

const XML_NS = "http://www.w3.org/XML/1998/namespace";
const XMLNS_NS = "http://www.w3.org/2000/xmlns/";

class Node {
	constructor(nodeType, nodeName) {
		this.nodeType = nodeType;
		this.nodeName = nodeName;
		this.parentNode = null;
		this.childNodes = [];
		this.ownerDocument = null;
	}
	get parentElement() {
		const p = this.parentNode;
		return p !== null && p.nodeType === 1 ? p : null;
	}
	get firstChild() {
		return this.childNodes[0] ?? null;
	}
	get lastChild() {
		return this.childNodes[this.childNodes.length - 1] ?? null;
	}
	_sibling(step, elements) {
		const siblings = this.parentNode?.childNodes;
		if (!siblings) return null;
		for (let i = siblings.indexOf(this) + step; i >= 0 && i < siblings.length; i += step) {
			if (!elements || siblings[i].nodeType === 1) return siblings[i];
		}
		return null;
	}
	get nextSibling() {
		return this._sibling(1, false);
	}
	get previousSibling() {
		return this._sibling(-1, false);
	}
	get nextElementSibling() {
		return this._sibling(1, true);
	}
	get previousElementSibling() {
		return this._sibling(-1, true);
	}
	get children() {
		return this.childNodes.filter((n) => n.nodeType === 1);
	}
	get childElementCount() {
		return this.children.length;
	}
	get firstElementChild() {
		return this.childNodes.find((n) => n.nodeType === 1) ?? null;
	}
	get lastElementChild() {
		const nodes = this.childNodes;
		for (let i = nodes.length - 1; i >= 0; i--) if (nodes[i].nodeType === 1) return nodes[i];
		return null;
	}
	hasChildNodes() {
		return this.childNodes.length !== 0;
	}
	get nodeValue() {
		return null;
	}
	get textContent() {
		const nodes = this.childNodes;
		if (nodes.length === 1) return nodes[0].textContent;
		let s = "";
		for (let i = 0; i < nodes.length; i++) if (nodes[i].nodeType !== 7) s += nodes[i].textContent;
		return s;
	}
	// The elements under this node, in document order, for which `test` is true (or the
	// first of them, with `first`).
	_descendants(test, first) {
		const out = [];
		const stack = [];
		for (let i = this.childNodes.length - 1; i >= 0; i--) stack.push(this.childNodes[i]);
		while (stack.length) {
			const n = stack.pop();
			if (n.nodeType !== 1) continue;
			if (test(n)) {
				if (first) return n;
				out.push(n);
			}
			for (let i = n.childNodes.length - 1; i >= 0; i--) stack.push(n.childNodes[i]);
		}
		return first ? null : out;
	}
	getElementsByTagName(name) {
		return this._descendants((e) => name === "*" || e.nodeName === name, false);
	}
	getElementsByTagNameNS(ns, localName) {
		return this._descendants((e) => (ns === "*" || e.namespaceURI === ns) && (localName === "*" || e.localName === localName),
			false);
	}
	querySelector(selectors) {
		const groups = parseSelectors(selectors);
		return this._descendants((e) => matchesSelectors(e, groups), true);
	}
	querySelectorAll(selectors) {
		const groups = parseSelectors(selectors);
		return this._descendants((e) => matchesSelectors(e, groups), false);
	}
	getElementById(id) {
		return this._descendants((e) => e.getAttribute("id") === id, true);
	}
}
for (const [name, value] of Object.entries({
	ELEMENT_NODE: 1, ATTRIBUTE_NODE: 2, TEXT_NODE: 3, CDATA_SECTION_NODE: 4, PROCESSING_INSTRUCTION_NODE: 7,
	COMMENT_NODE: 8, DOCUMENT_NODE: 9,
})) {
	Node[name] = value;
	Node.prototype[name] = value;
}

class Text extends Node {
	constructor(data) {
		super(3, "#text");
		this.data = data;
	}
	get nodeValue() {
		return this.data;
	}
	get textContent() {
		return this.data;
	}
	get wholeText() {
		return this.data;
	}
}

class ProcessingInstruction extends Node {
	constructor(target, data) {
		super(7, target);
		this.target = target;
		this.data = data;
	}
	get nodeValue() {
		return this.data;
	}
	get textContent() {
		return this.data;
	}
}

// "prefix:local" as [prefix or null, local]
function qualified(name) {
	const colon = name.indexOf(":");
	return colon < 0 ? [null, name] : [name.slice(0, colon), name.slice(colon + 1)];
}

// The namespace a prefix (null: the default one) stands for at `element`.
function lookupNamespace(element, prefix) {
	if (prefix === "xml") return XML_NS;
	if (prefix === "xmlns") return XMLNS_NS;
	const name = prefix === null ? "xmlns" : `xmlns:${prefix}`;
	for (let e = element; e !== null && e.nodeType === 1; e = e.parentNode) {
		const value = e.getAttribute(name);
		if (value !== null) return value || null;
	}
	return null;
}

class Attr {
	constructor(name, value, ownerElement) {
		this.name = name;
		this.value = value;
		this.ownerElement = ownerElement;
		[this.prefix, this.localName] = qualified(name);
	}
	get nodeType() {
		return 2;
	}
	get nodeName() {
		return this.name;
	}
	get nodeValue() {
		return this.value;
	}
	get textContent() {
		return this.value;
	}
	get namespaceURI() {
		if (this.name === "xmlns") return XMLNS_NS;
		return this.prefix === null ? null : lookupNamespace(this.ownerElement, this.prefix);
	}
}

// An element's attributes, by index and by name (a NamedNodeMap).
class Attributes extends Array {
	getNamedItem(name) {
		return this.find((a) => a.name === name) ?? null;
	}
	getNamedItemNS(ns, localName) {
		return this.find((a) => a.localName === localName && a.namespaceURI === ns) ?? null;
	}
	item(i) {
		return this[i] ?? null;
	}
}

class XMLElement extends Node {
	constructor(name) {
		super(1, name);
		this.tagName = name;
		[this.prefix, this.localName] = qualified(name);
		this.attributes = new Attributes();
		this._values = Object.create(null);
	}
	_setAttribute(name, value) {
		const attr = new Attr(name, value, this);
		this.attributes.push(attr);
		if (!(name in this.attributes)) this.attributes[name] = attr;
		this._values[name] = value;
	}
	getAttribute(name) {
		return this._values[name] ?? null;
	}
	hasAttribute(name) {
		return name in this._values;
	}
	hasAttributes() {
		return this.attributes.length !== 0;
	}
	getAttributeNames() {
		return this.attributes.map((a) => a.name);
	}
	getAttributeNS(ns, localName) {
		return this.attributes.getNamedItemNS(ns || null, localName)?.value ?? null;
	}
	hasAttributeNS(ns, localName) {
		return this.attributes.getNamedItemNS(ns || null, localName) !== null;
	}
	get id() {
		return this._values.id ?? "";
	}
	get namespaceURI() {
		return lookupNamespace(this, this.prefix);
	}
	matches(selectors) {
		return matchesSelectors(this, parseSelectors(selectors));
	}
	closest(selectors) {
		const groups = parseSelectors(selectors);
		for (let e = this; e !== null && e.nodeType === 1; e = e.parentNode) if (matchesSelectors(e, groups)) return e;
		return null;
	}
}

class XMLDocument extends Node {
	constructor(contentType = "application/xml") {
		super(9, "#document");
		this.contentType = contentType;
	}
	get documentElement() {
		return this.firstElementChild;
	}
}

// Selectors: lists (a, b) of compounds joined by descendant combinators (a b), each a type
// (or *) followed by any of #id, [attr], [attr=value] (quoted or not). That is all three's
// loaders use; anything else throws, as a selector a browser can't parse would.
const selectorCache = new Map();
const COMPOUND = /([\w-]+|\*)?((?:#[\w-]+|\[\s*[\w:-]+\s*(?:=\s*(?:"[^"]*"|'[^']*'|[\w-]+)\s*)?\])*)\s*/y;
const SELECTOR_PART = /#[\w-]+|\[(?:[^\]"']|"[^"]*"|'[^']*')*\]/g;
const ATTRIBUTE_SELECTOR = /^\[\s*([\w:-]+)\s*(?:=\s*(?:"([^"]*)"|'([^']*)'|([\w-]+))\s*)?\]$/;

// "a, b" as ["a", " b"], minding commas inside quotes
function splitSelectorList(source) {
	const out = [];
	let quote = null;
	let start = 0;
	for (let i = 0; i < source.length; i++) {
		const c = source[i];
		if (quote !== null) {
			if (c === quote) quote = null;
		} else if (c === '"' || c === "'") {
			quote = c;
		} else if (c === ",") {
			out.push(source.slice(start, i));
			start = i + 1;
		}
	}
	out.push(source.slice(start));
	return out;
}

function parseSelectors(source) {
	let groups = selectorCache.get(source);
	if (groups) return groups;
	const invalid = () => new DOMException(`'${source}' is not a valid selector`, "SyntaxError");
	groups = [];
	for (const group of splitSelectorList(source)) {
		const text = group.trim();
		const compounds = [];
		COMPOUND.lastIndex = 0;
		while (COMPOUND.lastIndex < text.length) {
			const m = COMPOUND.exec(text);
			if (!m || (m[1] === undefined && m[2] === "")) throw invalid();
			const compound = { tag: m[1] && m[1] !== "*" ? m[1] : null, attrs: [] };
			for (const part of m[2].match(SELECTOR_PART) ?? []) {
				if (part[0] === "#") {
					compound.attrs.push(["id", part.slice(1)]);
				} else {
					const a = ATTRIBUTE_SELECTOR.exec(part);
					compound.attrs.push([a[1], a[2] ?? a[3] ?? a[4]]);
				}
			}
			compounds.push(compound);
		}
		if (compounds.length === 0) throw invalid();
		groups.push(compounds);
	}
	selectorCache.set(source, groups);
	return groups;
}

function matchesCompound(e, compound) {
	if (compound.tag !== null && e.nodeName !== compound.tag) return false;
	for (const [name, value] of compound.attrs) {
		const v = e.getAttribute(name);
		if (v === null || (value !== undefined && v !== value)) return false;
	}
	return true;
}

// The last compound matches `e`, and each one before it an ancestor of the next.
function matchesSelectors(e, groups) {
	for (const compounds of groups) {
		let i = compounds.length - 1;
		if (!matchesCompound(e, compounds[i])) continue;
		for (let a = e.parentNode; i > 0 && a !== null && a.nodeType === 1; a = a.parentNode) {
			if (matchesCompound(a, compounds[i - 1])) i--;
		}
		if (i === 0) return true;
	}
	return false;
}

function appendNode(doc, parent, child) {
	child.parentNode = parent;
	child.ownerDocument = doc;
	parent.childNodes.push(child);
	return child;
}

globalThis.DOMParser = class DOMParser {
	// The fields of N.xmlParse's events (see src/xml.c3), into nodes.
	parseFromString(string, type = "application/xml") {
		const doc = new XMLDocument(type);
		const fields = N.xmlParse(String(string)).split("\0");
		let node = doc;
		for (let i = 0; i + 1 < fields.length; i += 2) {
			const head = fields[i];
			const value = fields[i + 1];
			switch (head[0]) {
				case "<":
					node = appendNode(doc, node, new XMLElement(head.slice(1)));
					break;
				case "=":
					node._setAttribute(head.slice(1), value);
					break;
				case ">":
					node = node.parentNode;
					break;
				case "#": {
					if (node === doc) break;
					const last = node.childNodes[node.childNodes.length - 1];
					if (last !== undefined && last.nodeType === 3) last.data += value;
					else appendNode(doc, node, new Text(value));
					break;
				}
				case "?":
					appendNode(doc, node, new ProcessingInstruction(head.slice(1), value));
					break;
				case "!": {
					const error = new XMLDocument(type);
					const div = appendNode(error, appendNode(error, error, new XMLElement("parsererror")), new XMLElement("div"));
					appendNode(error, div, new Text(head.slice(1)));
					return error;
				}
			}
		}
		return doc;
	}
};
globalThis.Node ??= Node;
globalThis.XMLDocument ??= XMLDocument;
