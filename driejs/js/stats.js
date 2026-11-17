// three's examples/jsm/libs/stats.module.js for driejs: the same API and the same
// numbers (frames per second over each second, milliseconds from begin() or the last
// update() to end(), each with its lowest and highest), drawn by the host over the picture
// instead of into a DOM canvas. The host maps "three/addons/libs/stats.module.js" here.
const N = globalThis.__drie;

class Panel {
	constructor(name, fg, bg) {
		this.name = name;
		this.dom = document.createElement("canvas");
		this.value = -1;
		this.min = Infinity;
		this.max = 0;
	}
	update(value) {
		this.value = value;
		this.min = Math.min(this.min, value);
		this.max = Math.max(this.max, value);
	}
}

export default class Stats {
	static Panel = Panel;

	constructor() {
		this.REVISION = 16;
		this.dom = document.createElement("div");
		this.domElement = this.dom;
		this._fps = new Panel("FPS", "#0ff", "#002");
		this._ms = new Panel("MS", "#0f0", "#020");
		this._begin = performance.now();
		this._prev = this._begin;
		this._frames = 0;
	}

	addPanel(panel) {
		return panel;
	}

	// The host shows both panels, so there is nothing to switch.
	showPanel() {}

	setMode() {}

	begin() {
		this._begin = performance.now();
	}

	end() {
		this._frames++;
		const time = performance.now();
		this._ms.update(time - this._begin);
		if (time >= this._prev + 1000) {
			this._fps.update((this._frames * 1000) / (time - this._prev));
			this._prev = time;
			this._frames = 0;
		}
		const f = this._fps;
		const m = this._ms;
		N.stats(f.value, f.min, f.max, m.value, m.min, m.max);
		return time;
	}

	update() {
		this._begin = this.end();
	}
}
