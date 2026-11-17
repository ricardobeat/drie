// three's examples/jsm/libs/stats.module.js for driejs: the same API and the same
// numbers (frames per second over each second, milliseconds from begin() or the last
// update() to end(), each with its lowest and highest), drawn by the host over the picture
// instead of into a DOM canvas. The host maps "three/addons/libs/stats.module.js" here.
const N = globalThis.__drie;

// How many seconds the lowest and highest values cover.
const WINDOW = 10;

// One value with its lowest and highest over the last WINDOW seconds, kept as one
// (min, max) pair per second: roll() closes the current second.
class Panel {
	constructor(name, fg, bg) {
		this.name = name;
		this.dom = document.createElement("canvas");
		this.value = -1;
		this.min = Infinity;
		this.max = 0;
		this._seconds = [];
		this._min = Infinity;
		this._max = 0;
	}
	update(value) {
		this.value = value;
		this._min = Math.min(this._min, value);
		this._max = Math.max(this._max, value);
		this.min = Math.min(this.min, value);
		this.max = Math.max(this.max, value);
	}
	roll() {
		if (this._min === Infinity) return;
		this._seconds.push([this._min, this._max]);
		if (this._seconds.length > WINDOW) this._seconds.shift();
		this._min = Infinity;
		this._max = 0;
		this.min = Infinity;
		this.max = 0;
		for (const [lo, hi] of this._seconds) {
			this.min = Math.min(this.min, lo);
			this.max = Math.max(this.max, hi);
		}
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
		// both clocks start at the first frame, not here: the time a script spends loading
		// after new Stats() is not a frame
		this._begin = -1;
		this._prev = -1;
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
		const time = performance.now();
		if (this._prev < 0) {
			this._prev = time;
			N.stats(0, 0, 0, 0, 0, 0);
			return time;
		}
		this._frames++;
		if (this._begin >= 0) this._ms.update(time - this._begin);
		if (time >= this._prev + 1000) {
			this._fps.update((this._frames * 1000) / (time - this._prev));
			this._fps.roll();
			this._ms.roll();
			this._prev = time;
			this._frames = 0;
		}
		const f = this._fps;
		const m = this._ms;
		const lo = (x) => (x === Infinity ? 0 : x);
		N.stats(f.value, lo(f.min), f.max, m.value, lo(m.min), m.max);
		return time;
	}

	update() {
		this._begin = this.end();
	}
}
