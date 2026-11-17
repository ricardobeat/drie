// WebAssembly on WAMR's fast interpreter (src/wasm.c3). What Emscripten's glue uses: compiling
// and instantiating with function imports, exported functions, memory (its buffer and grow) and
// tables of functions. Imported memories, tables and globals, exported globals and multiple
// results are not supported. i64 values cross as numbers, not BigInts.

const N = globalThis.__drie;

class CompileError extends Error {
	constructor(message) {
		super(message);
		this.name = "CompileError";
	}
}
class LinkError extends Error {
	constructor(message) {
		super(message);
		this.name = "LinkError";
	}
}
class RuntimeError extends Error {
	constructor(message) {
		super(message);
		this.name = "RuntimeError";
	}
}

N.wasmInit((kind, message) => (kind === "CompileError" ? new CompileError(message) : new RuntimeError(message)));

function bytesOf(source) {
	if (source instanceof ArrayBuffer) return new Uint8Array(source);
	if (ArrayBuffer.isView(source)) return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
	throw new TypeError("WebAssembly: expected an ArrayBuffer or a typed array");
}

function lines(text) {
	return text.split("\n").filter((l) => l.length).map((l) => l.split("\t"));
}

class Module {
	constructor(source) {
		this._handle = N.wasmCompile(bytesOf(source));
		this._imports = lines(N.wasmImports(this._handle)).map(([kind, module, name]) => ({ module, name, kind }));
		this._exports = lines(N.wasmExports(this._handle)).map(([kind, name]) => ({ name, kind }));
	}
	static imports(module) {
		return module._imports.map((i) => ({ ...i }));
	}
	static exports(module) {
		return module._exports.map((e) => ({ ...e }));
	}
	static customSections() {
		return [];
	}
}

// a wasm function as a JS function
function wrap(func) {
	const f = (...args) => N.wasmCall(func, ...args);
	Object.defineProperty(f, "length", { value: N.wasmArity(func) });
	return f;
}

class Memory {
	constructor() {
		throw new TypeError("WebAssembly.Memory: only exported memories are supported");
	}
	get buffer() {
		return N.wasmMemoryBuffer(this._instance, this._export);
	}
	grow(pages) {
		const old = N.wasmMemoryGrow(this._instance, this._export, pages >>> 0);
		if (old < 0) throw new RangeError("WebAssembly.Memory.grow: failed to grow memory");
		return old;
	}
}

class Table {
	constructor() {
		throw new TypeError("WebAssembly.Table: only exported tables are supported");
	}
	get length() {
		return N.wasmTableSize(this._instance, this._export);
	}
	get(index) {
		const func = N.wasmTableGet(this._instance, this._export, index >>> 0);
		if (func < 0) return null;
		let f = this._functions.get(func);
		if (!f) this._functions.set(func, (f = wrap(func)));
		return f;
	}
}

function exported(Class, instance, index) {
	const o = Object.create(Class.prototype);
	o._instance = instance;
	o._export = index;
	if (Class === Table) o._functions = new Map();
	return o;
}

class Instance {
	constructor(module, importObject = {}) {
		for (const { module: from, name, kind } of module._imports) {
			const value = importObject?.[from]?.[name];
			if (kind !== "function") throw new LinkError(`WebAssembly.Instance: import ${from}.${name}: only function imports are supported`);
			if (typeof value !== "function") throw new LinkError(`WebAssembly.Instance: import ${from}.${name} is not a function`);
			N.wasmImport(value);
		}
		const handle = N.wasmInstantiate(module._handle);
		const exports = Object.create(null);
		module._exports.forEach(({ name, kind }, i) => {
			if (kind === "function") exports[name] = wrap(N.wasmExportFunc(handle, i));
			else if (kind === "memory") exports[name] = exported(Memory, handle, i);
			else if (kind === "table") exports[name] = exported(Table, handle, i);
		});
		this.exports = Object.freeze(exports);
	}
}

function validate(source) {
	try {
		new Module(source);
		return true;
	} catch {
		return false;
	}
}

globalThis.WebAssembly = {
	Module,
	Instance,
	Memory,
	Table,
	CompileError,
	LinkError,
	RuntimeError,
	validate,
	compile: (source) => new Promise((resolve) => resolve(new Module(source))),
	instantiate(source, importObject) {
		return new Promise((resolve) => {
			if (source instanceof Module) resolve(new Instance(source, importObject));
			else {
				const module = new Module(source);
				resolve({ module, instance: new Instance(module, importObject) });
			}
		});
	},
};
