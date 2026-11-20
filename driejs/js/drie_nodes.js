// three's node materials (three/webgpu's MeshStandardNodeMaterial... with TSL graphs in
// colorNode, positionNode...) drawn by drie: each graph is rebuilt as drie's own (nodes.c3,
// through the node* natives), which drie prints into its shaders in the backend's language,
// as three's WebGPURenderer prints TSL as WGSL. three's NodeBuilder doesn't run: the graph is
// read as it is, and TSL's Fn bodies run here, a call at a time, to get their statements
// (toVar, assign, If, Loop, Break) and what they return.
//
// What carries over: the material slots colorNode, opacityNode, normalNode (in view space, as
// three's), roughnessNode, metalnessNode, emissiveNode, positionNode and outputNode; constants,
// uniforms (their .value read every frame, onFrameUpdate/onRenderUpdate run), textures (read
// in the fragment stage), the geometry's attributes (position, normal, uv, uv1, color),
// varyings, TSL's operators and math, swizzles, conversions, select, Fn calls (with layouts and
// overloads), variables, assignments, If/ElseIf/Else, Loop, Break, Continue, Discard, bumpMap,
// and the builtins: positionLocal/World/View, normalLocal/World/View (and their geometry's),
// cameraPosition, time, modelWorldMatrix, cameraViewMatrix, cameraProjectionMatrix,
// screenCoordinate, frontFacing, faceDirection, output. Anything else is an error naming it.
//
// Loaded by `three/webgpu` (src/main.c3): drie_renderer.js calls globalThis.__drieNodes.sync
// for a material whose isNodeMaterial is set.

import * as TSL from "../vendor/three.tsl.js";

const N = globalThis.__drie;

// drie's NodeType (nodes.c3), by three's names
const TYPES = ["float", "vec2", "vec3", "vec4", "bool", "mat4", "int", "ivec2", "ivec3", "ivec4", "uint", "uvec2", "uvec3", "uvec4", "void"];
const T = Object.fromEntries(TYPES.map((t, i) => [t, i]));
T.color = T.vec3;
const COMPONENTS = [1, 2, 3, 4, 1, 16, 1, 2, 3, 4, 1, 2, 3, 4, 0];
const scalarOf = (t) => (t <= 3 ? "float" : t === 4 ? "bool" : t === 5 ? "mat" : t <= 9 ? "int" : t <= 13 ? "uint" : "void");
const typeOf = (scalar, n) => (scalar === "bool" ? T.bool : { float: 0, int: 6, uint: 10 }[scalar] + n - 1);

// NodeBuiltin, NodeBinary, NodeFn, NodeLoop
const B = {
	UV: 0, UV1: 1, POSITION_LOCAL: 2, POSITION_WORLD: 3, NORMAL_LOCAL: 4, NORMAL_WORLD: 5, NORMAL_GEOMETRY: 6,
	CAMERA_POSITION: 7, VIEW_DIRECTION: 8, TIME: 9, VERTEX_COLOR: 10, FRAG_COORD: 11, FRONT_FACING: 12,
	MODEL_MATRIX: 13, VIEW_MATRIX: 14, PROJECTION_MATRIX: 15, OUTPUT: 16,
};
const BUILTIN_TYPES = [T.vec2, T.vec2, T.vec3, T.vec3, T.vec3, T.vec3, T.vec3, T.vec3, T.vec3, T.float, T.vec4, T.vec2, T.bool,
	T.mat4, T.mat4, T.mat4, T.vec4];
const OPS = {
	"+": 0, "-": 1, "*": 2, "/": 3, "<": 4, "<=": 5, ">": 6, ">=": 7, "==": 8, "!=": 9, "&&": 10, "||": 11, "%": 12, "&": 13,
	"|": 14, "^": 15, "<<": 16, ">>": 17, "^^": 18,
};
const FNS = [
	"sin", "cos", "tan", "asin", "acos", "atan", "exp", "exp2", "log", "log2", "sqrt", "inversesqrt", "abs", "sign", "floor",
	"ceil", "fract", "round", "trunc", "normalize", "length", "saturate", "dFdx", "dFdy", "fwidth", "pow", "min", "max", "step",
	"dot", "cross", "distance", "atan2", "reflect", "mod", "mix", "clamp", "smoothstep", "refract", "faceforward",
];
const FN = Object.fromEntries(FNS.map((f, i) => [f, i]));
const LOOP = { "<": 0, "<=": 1, ">": 2, ">=": 3, while: 4 };

// The slots of three's NodeMaterial that drie has (in materialNodes' order)
const SLOTS = ["colorNode", "opacityNode", "normalNode", "roughnessNode", "metalnessNode", "emissiveNode", "positionNode", "outputNode",
	"maskNode"];
// and those it hasn't
const UNSUPPORTED = ["fragmentNode", "vertexNode", "lightsNode", "envNode", "aoNode", "alphaTestNode", "maskShadowNode", "mrtNode",
	"castShadowNode", "castShadowPositionNode", "receivedShadowNode", "receivedShadowPositionNode", "shadowPositionNode",
	"depthNode", "backdropNode", "backdropAlphaNode", "clearcoatNode", "clearcoatRoughnessNode", "clearcoatNormalNode",
	"sheenNode", "sheenRoughnessNode", "iridescenceNode", "iridescenceIORNode", "iridescenceThicknessNode", "specularIntensityNode",
	"specularColorNode", "iorNode", "transmissionNode", "thicknessNode", "attenuationDistanceNode", "attenuationColorNode",
	"dispersionNode", "anisotropyNode", "geometryNode", "contextNode"];

class Unsupported extends Error {}

const unsupported = (what) => {
	throw new Unsupported(`driejs: ${what} isn't supported by drie's node materials`);
};

const nodeName = (n) => n?.constructor?.type ?? n?.constructor?.name ?? typeof n;
const isConditional = (n) => nodeName(n) === "ConditionalNode";

// Builtins by identity: TSL's exports, as drie nodes.
const BUILTINS = new Map();
const builtin = (name, make) => {
	const n = TSL[name];
	if (n !== undefined && n !== null) BUILTINS.set(n, make);
};
builtin("positionLocal", (t) => t.builtin(B.POSITION_LOCAL));
builtin("positionGeometry", (t) => t.builtin(B.POSITION_LOCAL));
builtin("positionWorld", (t) => t.builtin(B.POSITION_WORLD));
builtin("positionView", (t) => t.positionView());
builtin("positionViewDirection", (t) => t.call("normalize", t.mul(t.positionView(), t.num(-1))));
builtin("positionWorldDirection", (t) => t.call("normalize", t.sub(t.builtin(B.POSITION_WORLD), t.builtin(B.CAMERA_POSITION))));
builtin("normalLocal", (t) => t.builtin(B.NORMAL_LOCAL));
builtin("normalGeometry", (t) => t.builtin(B.NORMAL_LOCAL));
builtin("normalWorld", (t) => t.builtin(B.NORMAL_WORLD));
builtin("transformedNormalWorld", (t) => t.builtin(B.NORMAL_WORLD));
builtin("normalWorldGeometry", (t) => t.builtin(B.NORMAL_GEOMETRY));
builtin("normalView", (t) => t.toView(t.builtin(B.NORMAL_WORLD)));
builtin("transformedNormalView", (t) => t.toView(t.builtin(B.NORMAL_WORLD)));
builtin("normalViewGeometry", (t) => t.toView(t.builtin(B.NORMAL_GEOMETRY)));
builtin("cameraPosition", (t) => t.builtin(B.CAMERA_POSITION));
builtin("time", (t) => t.builtin(B.TIME));
builtin("modelWorldMatrix", (t) => t.builtin(B.MODEL_MATRIX));
builtin("cameraViewMatrix", (t) => t.builtin(B.VIEW_MATRIX));
builtin("cameraProjectionMatrix", (t) => t.builtin(B.PROJECTION_MATRIX));
builtin("modelViewMatrix", (t) => t.bin("*", t.builtin(B.VIEW_MATRIX), t.builtin(B.MODEL_MATRIX)));
builtin("screenCoordinate", (t) => t.builtin(B.FRAG_COORD));
builtin("frontFacing", (t) => t.builtin(B.FRONT_FACING));
builtin("faceDirection", (t) => N.nodeSelect(t.builtin(B.FRONT_FACING), t.num(1), t.num(-1)));
builtin("output", (t) => t.output());

// TSL functions drie does itself (by their Fn's body)
const FUNCTIONS = new Map();
const fn = (name, make) => {
	const f = TSL[name];
	if (f && f.shaderNode) FUNCTIONS.set(f.shaderNode, make);
};
// (the model's normal matrix as its rotation: for uniform scales)
fn("transformNormalToView", (t, [n]) => t.toView(t.call("normalize", t.swizzle(t.bin("*", t.builtin(B.MODEL_MATRIX), t.join(T.vec4, t.node(n), t.num(0))), "xyz"))));

// What a three type string is in drie's.
const typeId = (name) => {
	if (name === null || name === undefined) return undefined;
	const options = String(name).split("|");
	const t = T[options[0]];
	if (t === undefined) unsupported(`the type ${name}`);
	return t;
};

class Translator {
	constructor(renderer, material) {
		this.renderer = renderer;
		this.material = material;
		this.ids = new Map();       // three node: drie node
		this.types = new Map();     // drie node: its NodeType
		this.uniforms = [];         // { node, id, type, key }
		this.textures = [];         // [texture, drie id it had]
		this.once = new Map();
		this.uniformIds = new Map(); // three UniformNode: drie node (one each, whatever the context)
		this.uvShift = null;        // bumpMap's: textures read a pixel over ("dFdx") or down ("dFdy")
		this.block = null;          // the statements being made: { first, last }
		this.outputVar = 0;         // outputNode's `output` (see output())
		this.builder = new Proxy({
			material,
			shaderStage: "fragment",
			context: {},
			isAvailable: () => true,
		}, {
			get: (target, property) => {
				if (property in target) return target[property];
				if (typeof property === "symbol") return undefined;
				return () => unsupported(`three's node builder (${String(property)}), asked for by a TSL function`);
			},
		});
	}

	type(id) {
		let t = this.types.get(id);
		if (t === undefined) {
			t = N.nodeType(id);
			this.types.set(id, t);
		}
		return t;
	}

	// ---- making drie's nodes

	num(x, type = T.float) {
		return N.nodeConst(type, x, 0, 0, 0);
	}

	builtin(b) {
		return N.nodeBuiltin(b, BUILTIN_TYPES[b]);
	}

	bin(op, a, b) {
		return N.nodeBinary(OPS[op], a, b);
	}

	add(a, b) { return this.bin("+", a, b); }
	sub(a, b) { return this.bin("-", a, b); }
	mul(a, b) { return this.bin("*", a, b); }

	call(f, a, b = 0, c = 0) {
		return N.nodeCall(FN[f], a, b, c);
	}

	swizzle(a, components) {
		return N.nodeSwizzle(a, components);
	}

	join(type, ...parts) {
		return N.nodeJoin(type, ...parts);
	}

	convert(a, type) {
		return N.nodeConvert(a, type);
	}

	// A direction in the world into the camera's frame (and back, fromView).
	toView(n) {
		return this.call("normalize", this.swizzle(this.bin("*", this.builtin(B.VIEW_MATRIX), this.join(T.vec4, n, this.num(0))), "xyz"));
	}

	fromView(n) {
		// (the view matrix's rotation, transposed: the vector on the left)
		return this.call("normalize", this.swizzle(this.bin("*", this.join(T.vec4, n, this.num(0)), this.builtin(B.VIEW_MATRIX)), "xyz"));
	}

	positionView() {
		return this.swizzle(this.bin("*", this.builtin(B.VIEW_MATRIX), this.join(T.vec4, this.builtin(B.POSITION_WORLD), this.num(1))), "xyz");
	}

	// A statement, at the end of the block being made.
	emit(statement) {
		if (this.block === null) unsupported("a statement outside of a Fn (assign, If, Loop...)");
		const s = N.nodeAppend(this.block.last, statement);
		if (this.block.first === 0) this.block.first = s;
		this.block.last = s;
	}

	// Runs `run` (a TSL Fn's body, an If's, a Loop's) with a stack of its own, and makes the
	// statements it left there: { first: the block, value: what it returned (0: nothing) }.
	body(run) {
		const outerBlock = this.block;
		const outerStack = TSL.getCurrentStack();
		const stack = TSL.stack();
		TSL.setCurrentStack(stack);
		let out;
		try {
			out = run();
		} finally {
			// (nothing made from here on lands on a stack)
			TSL.setCurrentStack(null);
		}
		this.block = { first: 0, last: 0 };
		try {
			for (const s of stack.nodes) this.statement(s);
			let value = 0;
			if (out !== undefined && out !== null && out.isStackNode !== true) {
				const node = TSL.nodeObject(out);
				if (node && node.isNode) value = this.node(node);
			}
			return { first: this.block.first, value };
		} finally {
			this.block = outerBlock;
			TSL.setCurrentStack(outerStack);
		}
	}

	// A statement of a TSL stack.
	statement(n) {
		if (n.isVarNode) {
			this.emit(this.node(n));
		} else if (n.isAssignNode) {
			this.emit(N.nodeAssign(this.node(n.targetNode), this.node(n.sourceNode)));
		} else if (isConditional(n)) {
			this.emit(this.conditional(n));
		} else if (nodeName(n) === "LoopNode") {
			this.emit(this.loop(n));
		} else if (nodeName(n) === "ExpressionNode") {
			const jump = { break: 0, continue: 1, discard: 2 }[n.snippet];
			if (jump === undefined) unsupported(`the statement "${n.snippet}"`);
			this.emit(N.nodeJump(jump));
		} else if (n.isConstNode) {
			// (TSL stacks its constructors' constants: nothing to do)
		} else if (n.isShaderCallNodeInternal && this.isVoid(n)) {
			this.emit(this.node(n));
		} else if (this.ids.has(n)) {
			// (made already: as it was then)
		} else {
			// an assignable value (a constructor's, a function's result): its value here, kept
			const value = this.node(n);
			const type = this.type(value);
			if (type === T.void) {
				this.emit(value);
				return;
			}
			const kept = N.nodeVar(type, value);
			this.types.set(kept, type);
			this.ids.set(n, kept);
			this.emit(kept);
		}
	}

	isVoid(call) {
		return call.shaderNode?.nodeType === "void" || call.shaderNode?.layout?.type === "void";
	}

	// If (and ElseIf, Else): their bodies are ShaderNodes, run here.
	conditional(n) {
		const cond = this.convert(this.node(n.condNode), T.bool);
		const then = this.branch(n.ifNode);
		const otherwise = n.elseNode ? this.branch(n.elseNode) : 0;
		return N.nodeIf(cond, then, otherwise);
	}

	branch(b) {
		if (isConditional(b)) {
			// ElseIf: a block of one If
			const s = this.conditional(b);
			return N.nodeAppend(0, s);
		}
		if (typeof b.jsFunc === "function") {
			return this.body(() => (b.jsFunc.length > 0 ? b.jsFunc(this.builder) : b.jsFunc())).first;
		}
		// a node of its own (Discard(cond)'s expression)
		const saved = this.block;
		this.block = { first: 0, last: 0 };
		try {
			this.statement(b);
			return this.block.first;
		} finally {
			this.block = saved;
		}
	}

	// Loop(count, fn), Loop({ start, end, type, condition, name }, fn), Loop(a, b, fn) nested,
	// Loop(boolNode, fn) as a while.
	loop(n) {
		let params = n.params;
		if (typeof params[0] === "function") params = [TSL.bool(true), params[0]];
		const body = params[params.length - 1];
		const inputs = {};
		const loops = [];
		for (let i = 0; i < params.length - 1; i++) {
			const p = params[i];
			const name = (p.isNode !== true && p.name) || String.fromCharCode(105 + i);
			if (p.isNode === true) {
				const end = this.node(p);
				if (this.type(end) === T.bool) {
					loops.push({ how: LOOP.while, index: 0, from: 0, to: end });
					continue;
				}
				const index = N.nodeVar(T.int, 0);
				this.types.set(index, T.int);
				loops.push({ how: LOOP["<"], index, from: this.num(0, T.int), to: this.convert(end, T.int) });
				inputs[name] = this.placeholder(name, T.int, index);
				continue;
			}
			if (p.update !== undefined) unsupported("a Loop's update");
			const type = typeId(p.type || "int");
			const value = (x) => (typeof x === "number" ? this.num(x, type) : this.convert(this.node(TSL.nodeObject(x)), type));
			let from;
			let to;
			let condition = p.condition;
			if (p.start !== undefined && p.end === undefined) {
				from = this.sub(value(p.start), this.num(1, type));
				to = this.num(0, type);
				condition = ">=";
			} else {
				from = p.start === undefined ? this.num(0, type) : value(p.start);
				to = value(p.end);
				if (condition === undefined) {
					condition = typeof p.start === "number" && typeof p.end === "number" && p.start > p.end ? ">=" : "<";
				}
			}
			if (LOOP[condition] === undefined) unsupported(`a Loop's condition "${condition}"`);
			const index = N.nodeVar(type, 0);
			this.types.set(index, type);
			loops.push({ how: LOOP[condition], index, from, to });
			inputs[name] = this.placeholder(name, type, index);
		}
		let block = this.body(() => body(inputs)).first;
		for (let i = loops.length - 1; i >= 0; i--) {
			const l = loops[i];
			const s = N.nodeLoop(l.how, l.index, l.from, l.to, block);
			if (i === 0) return s;
			block = N.nodeAppend(0, s);
		}
		return 0;
	}

	// A three node standing for a drie one (a loop's index), for TSL code to use.
	placeholder(name, type, id) {
		const n = TSL.expression(name, TYPES[type]);
		this.ids.set(n, id);
		return n;
	}

	// ---- three's nodes, as drie's

	node(n) {
		if (n === null || n === undefined) unsupported("a missing node");
		if (typeof n === "number" || typeof n === "boolean") return this.constant(TSL.nodeObject(n));
		let id = this.ids.get(n);
		if (id !== undefined) return id;
		const make = BUILTINS.get(n);
		id = make ? make(this) : this.make(n);
		this.ids.set(n, id);
		return id;
	}

	make(n) {
		if (n.isConstNode) return this.constant(n);
		if (n.isTextureNode) return this.texture(n);
		if (n.isUniformNode) return this.uniform(n);
		if (n.isOperatorNode) return this.operator(n);
		if (n.isMathNode) return this.math(n);
		if (n.isSplitNode) return this.swizzle(this.node(n.node), n.components);
		if (n.isVarNode) {
			const init = n.node ? this.node(n.node) : 0;
			const type = init ? this.type(init) : typeId(n.nodeType ?? "float");
			const v = N.nodeVar(type, init);
			this.types.set(v, type);
			return v;
		}
		if (n.isAssignNode) {
			// an assignment's value: the target, assigned here
			const target = this.node(n.targetNode);
			this.emit(N.nodeAssign(target, this.node(n.sourceNode)));
			return target;
		}
		if (n.isShaderCallNodeInternal) return this.fnCall(n);
		if (isConditional(n)) {
			if (typeof n.ifNode?.jsFunc === "function") unsupported("If() as a value (use select())");
			const [a, b] = this.weak([n.ifNode, n.elseNode]);
			return N.nodeSelect(this.convert(this.node(n.condNode), T.bool), a, b);
		}
		switch (nodeName(n)) {
			case "ConvertNode": {
				const x = this.node(n.node);
				const options = String(n.convertTo).split("|").map(typeId);
				const c = COMPONENTS[this.type(x)];
				const to = options.find((t) => COMPONENTS[t] === c) ?? options[0];
				return this.convert(x, to);
			}
			case "JoinNode": {
				const parts = n.nodes.map((p) => this.node(p));
				let type = n.nodeType ? typeId(n.nodeType) : undefined;
				if (type === undefined) {
					const count = parts.reduce((s, p) => s + COMPONENTS[this.type(p)], 0);
					type = typeOf(scalarOf(this.type(parts[0])) === "bool" ? "float" : scalarOf(this.type(parts[0])), count);
				}
				if (parts.length > 4) unsupported("a vector of more than four parts");
				return this.join(type, ...parts);
			}
			case "AttributeNode": {
				const name = n._attributeName ?? n.attributeName;
				const b = { position: B.POSITION_LOCAL, normal: B.NORMAL_LOCAL, uv: B.UV, uv1: B.UV1, color: B.VERTEX_COLOR }[name];
				if (b === undefined) unsupported(`the geometry attribute "${name}"`);
				return this.builtin(b);
			}
			case "VertexColorNode":
				return this.builtin(B.VERTEX_COLOR);
			case "VaryingNode": {
				// a variable of the vertex stage (TSL assigns to varyings there), carried
				const init = this.node(n.node);
				const type = this.type(init);
				const v = N.nodeVar(type, init);
				this.types.set(v, type);
				return N.nodeVarying(v);
			}
			case "SubBuild":
			case "SubBuildNode":
			case "ContextNode":
			case "IsolateNode":
			case "CacheNode":
				return this.node(n.node);
			case "BypassNode":
				this.statement(n.callNode);
				return this.node(n.outputNode);
			case "FunctionOverloadingNode":
				return this.overloaded(n);
			case "BumpMapNode":
				return this.bumpMap(n);
			case "FrontFacingNode":
				return this.builtin(B.FRONT_FACING);
			case "PropertyNode":
				if (n.name === "Output" || n.name === "output") return this.output();
				unsupported(`the property "${n.name}"`);
				break;
			case "StackNode":
				if (n.outputNode) {
					const block = { first: 0, last: 0 };
					const saved = this.block;
					this.block = block;
					try {
						for (const s of n.nodes) this.statement(s);
						return N.nodeSeq(block.first, this.node(n.outputNode));
					} finally {
						this.block = saved;
					}
				}
				break;
		}
		unsupported(`the TSL node ${nodeName(n)}`);
	}

	constant(n) {
		const v = n.value;
		let type = n.nodeType ? typeId(n.nodeType) : undefined;
		if (typeof v === "number") {
			type = type ?? T.float;
			if (COMPONENTS[type] !== 1) return this.convert(this.num(v), type);
			return this.num(v, type);
		}
		if (typeof v === "boolean") return this.num(v ? 1 : 0, T.bool);
		if (v && v.isColor) return N.nodeConst(T.vec3, v.r, v.g, v.b, 0);
		if (v && (v.isVector2 || v.isVector3 || v.isVector4)) {
			const c = v.isVector2 ? 2 : v.isVector3 ? 3 : 4;
			type = type !== undefined && COMPONENTS[type] === c ? type : typeOf("float", c);
			return N.nodeConst(type, v.x, v.y, v.z ?? 0, v.w ?? 0);
		}
		if (v && v.isMatrix4) {
			const u = N.nodeUniform(T.mat4);
			N.nodeSet(u, new Float32Array(v.elements));
			return u;
		}
		unsupported(`a constant ${v && v.constructor ? v.constructor.name : typeof v}`);
	}

	// A number made weak by TSL (`x.add(1)`) takes the kind of the other operands' (an
	// integer's, or a float's).
	weak(nodes) {
		const isWeak = (n) => n && n.isConstNode && n.isWeak && typeof n.value === "number";
		const ids = nodes.map((n) => (n === null || n === undefined || isWeak(n) ? 0 : this.node(n)));
		let kind = "float";
		for (const id of ids) {
			if (id) {
				const s = scalarOf(this.type(id));
				if (s === "int" || s === "uint") kind = s;
				break;
			}
		}
		return nodes.map((n, i) => {
			if (ids[i] || n === null || n === undefined) return ids[i];
			const k = kind !== "float" && Number.isInteger(n.value) ? kind : "float";
			return this.num(n.value, typeOf(k, 1));
		});
	}

	// TSL's vectors of different sizes in one operation: all as the widest (three's
	// OperatorNode and MathNode build their inputs so: a vec3 gets a w of 1, a vec4 loses it).
	widest(ids) {
		let n = 1;
		for (const id of ids) {
			if (!id) continue;
			const t = this.type(id);
			if (t !== T.mat4 && COMPONENTS[t] > n) n = COMPONENTS[t];
		}
		return ids.map((id) => {
			if (!id) return id;
			const t = this.type(id);
			if (t === T.mat4 || COMPONENTS[t] === 1 || COMPONENTS[t] === n) return id;
			const s = scalarOf(t);
			return this.convert(id, typeOf(s === "bool" ? "float" : s, n));
		});
	}

	operator(n) {
		const op = n.op;
		if (op === "!" || op === "~") return N.nodeNot(this.node(n.aNode), op === "~");
		if (OPS[op] === undefined) unsupported(`the operator ${op}`);
		let [a, b] = this.weak([n.aNode, n.bNode]);
		if (OPS[op] < OPS["<"] || OPS[op] >= OPS["%"]) [a, b] = this.widest([a, b]);
		return this.bin(op, a, b);
	}

	math(n) {
		const m = n.method;
		let [a, b, c] = this.weak([n.aNode, n.bNode, n.cNode]);
		if (m !== "transformDirection" && m !== "cross") [a, b, c] = this.widest([a, b, c]);
		const kindOf = (x) => scalarOf(this.type(x));
		const one = (x) => this.num(1, typeOf(kindOf(x) === "bool" ? "float" : kindOf(x), 1));
		switch (m) {
			case "atan": return b ? this.call("atan2", a, b) : this.call("atan", a);
			case "negate": return this.mul(a, this.num(-1, typeOf(kindOf(a) === "uint" ? "int" : kindOf(a) === "bool" ? "float" : kindOf(a), 1)));
			case "oneMinus": return this.sub(one(a), a);
			case "reciprocal": return this.bin("/", this.num(1), a);
			case "difference": return this.call("abs", this.sub(a, b));
			case "equals": return this.bin("==", a, b);
			case "radians": return this.mul(a, this.num(Math.PI / 180));
			case "degrees": return this.mul(a, this.num(180 / Math.PI));
			case "transformDirection": return this.call("normalize", this.swizzle(this.bin("*", b, this.join(T.vec4, a, this.num(0))), "xyz"));
			case "sinh": return this.mul(this.sub(this.call("exp", a), this.call("exp", this.mul(a, this.num(-1)))), this.num(0.5));
			case "cosh": return this.mul(this.add(this.call("exp", a), this.call("exp", this.mul(a, this.num(-1)))), this.num(0.5));
			case "tanh": {
				const e = this.call("exp", this.mul(a, this.num(2)));
				return this.bin("/", this.sub(e, this.num(1)), this.add(e, this.num(1)));
			}
		}
		if (FN[m] === undefined) unsupported(`the TSL function ${m}()`);
		return this.call(m, a, b, c);
	}

	uniform(n) {
		const made = this.uniformIds.get(n);
		if (made !== undefined) return made;
		let type = n.nodeType ? typeId(n.nodeType) : undefined;
		const v = n.value;
		if (type === undefined) {
			if (typeof v === "number") type = T.float;
			else if (typeof v === "boolean") type = T.bool;
			else if (v && v.isColor) type = T.vec3;
			else if (v && v.isVector2) type = T.vec2;
			else if (v && v.isVector3) type = T.vec3;
			else if (v && v.isVector4) type = T.vec4;
			else if (v && v.isMatrix4) type = T.mat4;
			else unsupported(`a uniform of ${v && v.constructor ? v.constructor.name : typeof v}`);
		}
		const id = N.nodeUniform(type);
		this.types.set(id, type);
		const u = { node: n, id, type, key: undefined };
		this.uniformIds.set(n, id);
		this.uniforms.push(u);
		setUniform(u);
		return id;
	}

	texture(n) {
		if (n.isCubeTextureNode || n.isTexture3DNode || n.compareNode || n.depthNode || n.gradNode || n.offsetNode) {
			unsupported(`this kind of texture node (${nodeName(n)})`);
		}
		const texture = n.value;
		const id = this.renderer._texture(texture);
		this.textures.push([texture, id]);
		let at = n.uvNode ? this.convert(this.node(n.uvNode), T.vec2) : 0;
		if (this.uvShift) {
			if (!at) at = this.builtin(B.UV);
			at = this.add(at, this.call(this.uvShift, at));
		}
		const level = n.levelNode ? this.convert(this.node(n.levelNode), T.float) : 0;
		return N.nodeTexture(id, at, level);
	}

	// A call of a TSL Fn: its body run here, with the call's arguments.
	fnCall(n) {
		const shader = n.shaderNode;
		const own = FUNCTIONS.get(shader);
		if (own) return own(this, n.rawInputs ?? []);
		if (shader.once) {
			const done = this.once.get(shader);
			if (done !== undefined) return done;
		}
		const layout = shader.layout;
		const raw = n.rawInputs;
		let inputs;
		if (raw && raw.length === 1 && raw[0] && typeof raw[0] === "object" && raw[0].isNode !== true && !Array.isArray(raw[0])) {
			// named parameters: Fn(({ a, b }) => ...)({ a, b })
			inputs = [raw[0]];
			for (const k in raw[0]) {
				let x = TSL.nodeObject(raw[0][k]);
				const p = layout?.inputs?.find((i) => i.name === k);
				if (p && x && x.isNode) x = this.typed(x, p.type);
				inputs[k] = x;
			}
		} else {
			inputs = (raw ?? []).map((x, i) => {
				const node = TSL.nodeObject(x);
				const p = layout?.inputs?.[i];
				return p && node && node.isNode ? this.typed(node, p.type) : node;
			});
		}
		const jsFunc = shader.jsFunc;
		const hasParameters = raw !== null && raw !== undefined && raw.length > 0;
		const { first, value } = this.body(() => (hasParameters || jsFunc.length > 1 ? jsFunc(inputs, this.builder) : jsFunc(this.builder)));
		let result = value;
		const returns = layout?.type ?? shader.nodeType;
		if (result && returns && returns !== "void") result = this.convert(result, typeId(returns));
		const id = first === 0 && result ? result : N.nodeSeq(first, result);
		if (shader.once) this.once.set(shader, id);
		return id;
	}

	// `node` converted to the three type `type` (a Fn layout's parameter).
	typed(node, type) {
		const t = typeId(type);
		const id = this.node(node);
		if (this.type(id) === t) return node;
		const c = this.convert(id, t);
		return this.placeholder("arg", t, c);
	}

	// overloadingFn([...]): the function whose layout takes these arguments' types.
	overloaded(n) {
		const args = n.parametersNodes;
		const types = args.map((a) => TYPES[this.type(this.node(a))]);
		let best = null;
		for (const f of n.functionNodes) {
			const inputs = f.shaderNode?.layout?.inputs ?? [];
			if (inputs.length !== args.length) continue;
			if (inputs.every((p, i) => p.type === types[i] || (p.type === "color" && types[i] === "vec3"))) {
				best = f;
				break;
			}
			if (best === null) best = f;
		}
		if (best === null) unsupported(`an overloaded function without one for (${types.join(", ")})`);
		return this.node(best.shaderNode.call(args));
	}

	// `n` again, with its textures read a pixel over: three's forward differences
	shifted(n, shift) {
		const ids = this.ids;
		const once = this.once;
		this.ids = new Map();
		this.once = new Map();
		this.uvShift = shift;
		try {
			return this.convert(this.node(n), T.float);
		} finally {
			this.ids = ids;
			this.once = once;
			this.uvShift = null;
		}
	}

	// bumpMap(height, scale): three's dHdxy_fwd and perturbNormalArb
	bumpMap(n) {
		const scale = n.scaleNode ? this.convert(this.node(n.scaleNode), T.float) : this.num(1);
		const h = this.convert(this.node(n.textureNode), T.float);
		const dx = this.mul(this.sub(this.shifted(n.textureNode, "dFdx"), h), scale);
		const dy = this.mul(this.sub(this.shifted(n.textureNode, "dFdy"), h), scale);
		const pos = this.positionView();
		const normal = n.normalNode ? this.node(n.normalNode) : this.toView(this.builtin(B.NORMAL_WORLD));
		const sx = this.call("normalize", this.call("dFdx", pos));
		const sy = this.call("normalize", this.call("dFdy", pos));
		const r1 = this.call("cross", sy, normal);
		const r2 = this.call("cross", normal, sx);
		const face = N.nodeSelect(this.builtin(B.FRONT_FACING), this.num(1), this.num(-1));
		const det = this.mul(this.call("dot", sx, r1), face);
		const grad = this.mul(this.call("sign", det), this.add(this.mul(dx, r1), this.mul(dy, r2)));
		return this.call("normalize", this.sub(this.mul(this.call("abs", det), normal), grad));
	}

	// A material slot's graph.
	// outputNode's `output`: a variable that starts as what the light made, so that a Fn can
	// assign it (declared first in the slot's block)
	output() {
		if (this.outputVar === 0) unsupported("output outside of outputNode");
		return this.outputVar;
	}

	slot(name, n) {
		this.block = { first: 0, last: 0 };
		TSL.setCurrentStack(null);
		this.outputVar = 0;
		if (name === "outputNode") {
			this.outputVar = N.nodeVar(T.vec4, this.builtin(B.OUTPUT));
			this.types.set(this.outputVar, T.vec4);
			this.emit(this.outputVar);
		}
		let id = this.node(n);
		if (this.block.first !== 0) id = N.nodeSeq(this.block.first, id);
		this.block = null;
		if (name === "normalNode") id = this.fromView(this.convert(id, T.vec3));
		return id;
	}
}

// A uniform's value into drie, when it changed.
function setUniform(u) {
	const v = u.node.value;
	if (typeof v === "number" || typeof v === "boolean") {
		const x = Number(v);
		if (u.key === x) return;
		u.key = x;
		N.nodeSet(u.id, x, 0, 0, 0);
	} else if (v && v.isColor) {
		const key = `${v.r} ${v.g} ${v.b}`;
		if (u.key === key) return;
		u.key = key;
		N.nodeSet(u.id, v.r, v.g, v.b, 0);
	} else if (v && (v.isVector2 || v.isVector3 || v.isVector4)) {
		const key = `${v.x} ${v.y} ${v.z} ${v.w}`;
		if (u.key === key) return;
		u.key = key;
		N.nodeSet(u.id, v.x, v.y, v.z ?? 0, v.w ?? 0);
	} else if (v && v.isMatrix4) {
		const key = v.elements.join(" ");
		if (u.key === key) return;
		u.key = key;
		N.nodeSet(u.id, new Float32Array(v.elements));
	}
}

const warned = new WeakSet();

// The node material's graphs into drie's material, rebuilt when a slot changes (or a texture
// it reads arrives); its uniforms' values every frame.
function sync(renderer, material, frame) {
	const slots = SLOTS.map((s) => material[s] ?? null);
	let state = material.__drieNodes;
	const stale = state === undefined || state.material !== material.__drie || slots.some((s, i) => s !== state.slots[i])
		|| state.textures.some(([t, id]) => renderer._texture(t) !== id);
	if (stale) {
		if (!warned.has(material)) {
			warned.add(material);
			for (const s of UNSUPPORTED) {
				if (material[s]) console.warn(`driejs: ${material.type}.${s} isn't supported by drie (left out)`);
			}
		}
		const t = new Translator(renderer, material);
		const ids = SLOTS.map((s, i) => (slots[i] && slots[i].isNode ? t.slot(s, slots[i]) : 0));
		N.materialNodes(material.__drie, ...ids);
		state = { material: material.__drie, slots, textures: t.textures, uniforms: t.uniforms };
		material.__drieNodes = state;
	}
	for (const u of state.uniforms) {
		const n = u.node;
		if (n.updateType && n.updateType !== "none" && typeof n.update === "function") {
			n.update({ time: frame.time, deltaTime: frame.deltaTime, frameId: frame.id, renderer, material, camera: frame.camera, scene: frame.scene });
		}
		setUniform(u);
	}
}

globalThis.__drieNodes = { sync };
