// One scene, timed three ways: COUNT cubes in groups of 10, every cube turning, the camera
// circling. It keeps to the part of three.js's API that drie-js implements too, so this file
// runs as it is on three.js (driejs) and on drie-js; bench/cubes.c3 is the same scene in
// C3 on drie. See bench/cubes.sh.
import * as THREE from "three";

const WIDTH = 1280;
const HEIGHT = 720;
const COUNT = 5000;

const renderer = new THREE.WebGLRenderer({ antialias: true, width: WIDTH, height: HEIGHT, title: "cubes" });
renderer.setSize(WIDTH, HEIGHT);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, WIDTH / HEIGHT, 0.1, 500);

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const sun = new THREE.DirectionalLight(0xffffff, 2);
sun.position.set(5, 8, 3);
scene.add(sun);

const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshStandardMaterial({ color: 0x44aa88, roughness: 0.5 });
const cubes = [];
const side = Math.ceil(Math.sqrt(COUNT / 10));
for (let g = 0; g * 10 < COUNT; g++) {
	const group = new THREE.Group();
	group.position.set((g % side) * 4 - side * 2, 0, Math.floor(g / side) * 4 - side * 2);
	scene.add(group);
	for (let i = 0; i < 10 && g * 10 + i < COUNT; i++) {
		const cube = new THREE.Mesh(geometry, material);
		cube.position.set((i % 3) - 1, Math.floor(i / 3) * 0.5, 0);
		cube.scale.set(0.3, 0.3, 0.3);
		group.add(cube);
		cubes.push(cube);
	}
}

let frame = 0;
renderer.setAnimationLoop(() => {
	const t = frame++ / 60;
	for (let i = 0; i < cubes.length; i++) cubes[i].rotation.y = t + i;
	camera.position.set(Math.cos(t * 0.2) * side * 3, side * 2, Math.sin(t * 0.2) * side * 3);
	camera.lookAt(0, 0, 0);
	renderer.render(scene, camera);
});
