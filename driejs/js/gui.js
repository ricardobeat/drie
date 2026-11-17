// lil-gui's API with nothing on screen, for the examples' settings panels: every call works
// and returns what lil-gui returns (controllers and folders chain), and values keep what the
// example set. The host maps "three/addons/libs/lil-gui.module.min.js" here.
class Controller {
	constructor(parent, object, property) {
		this.parent = parent;
		this.object = object;
		this.property = property;
		this.initialValue = object?.[property];
	}
	getValue() {
		return this.object[this.property];
	}
	setValue(value) {
		this.object[this.property] = value;
		this._onChange?.call(this, value);
		this._onFinishChange?.call(this, value);
		return this;
	}
	onChange(fn) {
		this._onChange = fn;
		return this;
	}
	onFinishChange(fn) {
		this._onFinishChange = fn;
		return this;
	}
	reset() {
		return this.setValue(this.initialValue);
	}
	updateDisplay() {
		return this;
	}
	destroy() {}
}

for (const name of ["name", "min", "max", "step", "decimals", "listen", "options", "disable", "enable", "show", "hide"]) {
	Controller.prototype[name] = function () {
		return this;
	};
}

export class GUI {
	constructor(options = {}) {
		this.parent = options.parent;
		this.children = [];
		this.folders = [];
		this.controllers = [];
		this.domElement = document.createElement("div");
		this._title = options.title ?? "Controls";
	}
	add(object, property) {
		const c = new Controller(this, object, property);
		this.controllers.push(c);
		this.children.push(c);
		return c;
	}
	addColor(object, property) {
		return this.add(object, property);
	}
	addFolder(title) {
		const folder = new GUI({ parent: this, title });
		this.folders.push(folder);
		this.children.push(folder);
		return folder;
	}
	controllersRecursive() {
		return [...this.controllers, ...this.folders.flatMap((f) => f.controllersRecursive())];
	}
	foldersRecursive() {
		return [...this.folders, ...this.folders.flatMap((f) => f.foldersRecursive())];
	}
	title(title) {
		this._title = title;
		return this;
	}
	onChange() {
		return this;
	}
	onFinishChange() {
		return this;
	}
	save() {
		return {};
	}
	load() {
		return this;
	}
	reset() {
		return this;
	}
	destroy() {}
}

for (const name of ["open", "close", "show", "hide", "openAnimated"]) {
	GUI.prototype[name] = function () {
		return this;
	};
}

export default GUI;
