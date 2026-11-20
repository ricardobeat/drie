// three's RectAreaLightUniformsLib (three/addons/lights/RectAreaLightUniformsLib.js): for
// WebGLRenderer it puts the LTC tables into UniformsLib, which three.core.js does not have.
// drie has the tables itself (generated/ltc.c3) and hands them to its shader when a scene has
// a RectAreaLight, so init() has nothing to do.
export class RectAreaLightUniformsLib {
	static init() {}
}
