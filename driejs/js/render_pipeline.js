// three's RenderPipeline (three/webgpu) for driejs: walks `outputNode` to find the scene
// PassNode (pass, ssaaPass, toonOutlinePass) and post-processing nodes (bloom), and runs
// them through drie's renderer, OutlineEffect, or native UnrealBloomPass + OutputPass.
import { Color } from "../vendor/three.core.js";
import { OutlineEffect } from "./outline_effect.js";

function readNumber(n, fallback) {
	if (typeof n === "number") return n;
	if (n && typeof n.value === "number") return n.value;
	return fallback;
}

function readColor(n, fallback) {
	if (n && n.isColor) return n;
	if (n && n.value && n.value.isColor) return n.value;
	return fallback;
}

function inspectGraph(root) {
	let passNode = null;
	let toonNode = null;
	let bloomNode = null;
	const seen = new Set();
	const visit = (n) => {
		if (!n || typeof n !== "object" || seen.has(n)) return;
		seen.add(n);
		if (n.isToonOutlinePassNode || n.constructor?.name === "ToonOutlinePassNode") {
			toonNode = n;
			passNode = n;
			return;
		}
		if (n.isBloomNode || n.constructor?.name === "BloomNode") {
			bloomNode = n;
		}
		if (!passNode && n.scene && n.camera && (n.isPassNode || n.isSSAAPassNode || n.constructor?.name?.includes("PassNode"))) {
			passNode = n;
		}
		// Walk common TSL child node fields
		if (n.textureNode) visit(n.textureNode);
		if (n.inputNode) visit(n.inputNode);
		if (n.passNode) visit(n.passNode);
		if (n.node) visit(n.node);
		if (n.aNode) visit(n.aNode);
		if (n.bNode) visit(n.bNode);
		if (n.cNode) visit(n.cNode);
		if (n.sourceNode) visit(n.sourceNode);
		if (n.outputNode) visit(n.outputNode);
		if (n.condNode) visit(n.condNode);
		if (n.ifNode) visit(n.ifNode);
		if (n.elseNode) visit(n.elseNode);
		if (Array.isArray(n.nodes)) for (const c of n.nodes) visit(c);
		if (Array.isArray(n.rawInputs)) for (const c of n.rawInputs) visit(c);
		if (n.parameters && typeof n.parameters === "object") {
			for (const k in n.parameters) visit(n.parameters[k]);
		}
	};
	visit(root);
	return { passNode, toonNode, bloomNode };
}

export class RenderPipeline {
	constructor(renderer, outputNode = null) {
		this.isRenderPipeline = true;
		this.renderer = renderer;
		this.outputNode = outputNode;
		this.outputColorTransform = true;
		this.needsUpdate = true;
		this.context = {};
		this._outline = null;
	}

	render() {
		const renderer = this.renderer;
		if (!renderer || !this.outputNode) return;
		const { passNode, toonNode, bloomNode } = inspectGraph(this.outputNode);
		if (!passNode || !passNode.scene || !passNode.camera) return;
		const { scene, camera } = passNode;

		if (toonNode) {
			if (!this._outline) this._outline = new OutlineEffect(renderer);
			const col = readColor(toonNode.colorNode, null);
			if (col) this._outline.defaultColor.copy(col);
			else this._outline.defaultColor.setHex(0x000000);
			this._outline.defaultThickness = readNumber(toonNode.thicknessNode, 0.003);
			this._outline.defaultAlpha = readNumber(toonNode.alphaNode, 1.0);
			this._outline.render(scene, camera);
			return;
		}

		if (bloomNode && typeof renderer._renderBloom === "function") {
			const strength = readNumber(bloomNode.strength, 1.0);
			const radius = readNumber(bloomNode.radius, 0.0);
			const threshold = readNumber(bloomNode.threshold, 0.0);
			renderer._renderBloom(scene, camera, strength, radius, threshold);
			return;
		}

		renderer.render(scene, camera);
	}

	async renderAsync() {
		this.render();
	}

	dispose() {}
}
