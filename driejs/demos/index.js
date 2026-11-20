// The demos in this folder, as a page of thumbnails: click one to load its script in this
// one's place (the same process and window), Escape there comes back here. The wheel (or the arrow keys) scrolls.
//
//   just thumbs                  renders each demo's thumbnail into demos/thumbs/
//   just run demos/index.js
//
// A demo without a thumbnail shows as a grey tile. Each thumbnail holds the pipelines its
// demo drew with (a pipeline cache): the page has them all compiled in the background while it
// is up, with a bar in its top right corner till they are, so that a demo opens without the
// second or two its shaders would take. This script is driejs's own (not one of three.js's
// examples): it lists the folder, writes the names, loads the caches and starts the demos
// through the host (globalThis.__drie: files, label, labelScale, loadPipelineCache,
// pipelineCacheProgress, run).
import * as THREE from 'three';

const host = globalThis.__drie;

const names = host.files( host.baseDir ).split( '\n' )
	.filter( ( f ) => f.endsWith( '.js' ) && f !== 'index.js' )
	.map( ( f ) => f.slice( 0, - 3 ) )
	.sort();

const width = window.innerWidth;
const height = window.innerHeight;

// the page, in window points from its top left corner
const MARGIN = 16;
const COLUMNS = 4;
const glyph = 8 * host.labelScale();
const HEADER = MARGIN + glyph * 2;
const tileWidth = ( width - MARGIN * ( COLUMNS + 1 ) ) / COLUMNS;
const tileHeight = tileWidth * 9 / 16;
const rowHeight = tileHeight + glyph * 2 + MARGIN;
const rows = Math.ceil( names.length / COLUMNS );
const pageHeight = HEADER + rows * rowHeight + MARGIN;

const renderer = new THREE.WebGLRenderer( { antialias: true } );
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( width, height );
document.body.appendChild( renderer.domElement );

const scene = new THREE.Scene();
scene.background = new THREE.Color( 0x16181d );

// y down the page is -y in the scene; the camera as far back as shows the page's plane a
// window point to the unit (driejs has no OrthographicCamera)
const FOV = 30;
const distance = height / 2 / Math.tan( THREE.MathUtils.degToRad( FOV / 2 ) );
const camera = new THREE.PerspectiveCamera( FOV, width / height, distance / 2, distance * 2 );
camera.position.set( width / 2, - height / 2, distance );

const plane = new THREE.PlaneGeometry( 1, 1 );
const frameMaterial = new THREE.MeshBasicMaterial( { color: 0xffd866 } );
const loader = new THREE.TextureLoader();

const tiles = names.map( ( name, i ) => {

	const x = MARGIN + ( i % COLUMNS ) * ( tileWidth + MARGIN );
	const y = HEADER + Math.floor( i / COLUMNS ) * rowHeight;

	const material = new THREE.MeshBasicMaterial( { color: 0x30343c } );
	const thumbnail = 'thumbs/' + name + '.png';

	if ( host.fileSize( host.baseDir + thumbnail ) > 0 ) {

		host.loadPipelineCache( host.baseDir + thumbnail );

		loader.load( thumbnail, ( texture ) => {

			texture.colorSpace = THREE.SRGBColorSpace;
			material.map = texture;
			material.color.set( 0xffffff );
			material.needsUpdate = true;

		} );

	}

	const picture = new THREE.Mesh( plane, material );
	picture.scale.set( tileWidth, tileHeight, 1 );
	picture.position.set( x + tileWidth / 2, - ( y + tileHeight / 2 ), 0 );
	scene.add( picture );

	// a border, shown under the pointer
	const frame = new THREE.Mesh( plane, frameMaterial );
	frame.scale.set( tileWidth + 6, tileHeight + 6, 1 );
	frame.position.set( picture.position.x, picture.position.y, - 1 );
	frame.visible = false;
	scene.add( frame );

	return { name, x, y, frame };

} );

// the pipelines' progress: a bar in a box with a 2 point border, at the top right
const BAR_WIDTH = 160;
const BAR_HEIGHT = 10;
const BORDER = 2;
const barX = width - MARGIN - BAR_WIDTH;
const barY = MARGIN;

function rectangle( color, x, y, w, h, z ) {

	const mesh = new THREE.Mesh( plane, new THREE.MeshBasicMaterial( { color } ) );
	mesh.scale.set( w, h, 1 );
	mesh.position.set( x + w / 2, - ( y + h / 2 ), z );
	scene.add( mesh );
	return mesh;

}

const barBorder = rectangle( 0xa0a5af, barX - BORDER, barY - BORDER, BAR_WIDTH + BORDER * 2, BAR_HEIGHT + BORDER * 2, 1 );
const barInside = rectangle( 0x16181d, barX, barY, BAR_WIDTH, BAR_HEIGHT, 2 );
const bar = rectangle( 0xa0a5af, barX, barY, BAR_WIDTH, BAR_HEIGHT, 3 );
const progressBar = [ barBorder, barInside, bar ];

let scroll = 0;
let hovered = null;
const pointer = { x: - 1, y: - 1 };

function scrollBy( dy ) {

	scroll = Math.max( 0, Math.min( pageHeight - height, scroll + dy ) );

}

function tileAt( x, y ) {

	y += scroll;
	return tiles.find( ( t ) => x >= t.x && x < t.x + tileWidth && y >= t.y && y < t.y + tileHeight ) ?? null;

}

window.addEventListener( 'pointermove', ( event ) => {

	pointer.x = event.clientX;
	pointer.y = event.clientY;

} );

window.addEventListener( 'wheel', ( event ) => scrollBy( event.deltaY ) );

window.addEventListener( 'keydown', ( event ) => {

	if ( event.code === 'ArrowDown' || event.code === 'PageDown' ) scrollBy( rowHeight );
	if ( event.code === 'ArrowUp' || event.code === 'PageUp' ) scrollBy( - rowHeight );

} );

window.addEventListener( 'click', ( event ) => {

	const tile = tileAt( event.clientX, event.clientY );
	if ( tile ) host.run( host.baseDir + tile.name + '.js' );

} );

renderer.setAnimationLoop( () => {

	hovered = tileAt( pointer.x, pointer.y );
	camera.position.y = - height / 2 - scroll;

	// (it stays put as the page scrolls; gone once all have compiled)
	const progress = host.pipelineCacheProgress();
	for ( const mesh of progressBar ) {

		mesh.visible = progress < 1;
		mesh.position.y = - ( mesh === barBorder ? barY - BORDER : barY ) - mesh.scale.y / 2 - scroll;

	}

	bar.scale.x = BAR_WIDTH * progress;
	bar.position.x = barX + bar.scale.x / 2;

	const columns = Math.floor( tileWidth / glyph );

	host.label( 'driejs demos: click one to run it, Escape comes back', MARGIN, MARGIN - scroll, 160, 165, 175 );

	for ( const tile of tiles ) {

		tile.frame.visible = tile === hovered;
		const y = tile.y + tileHeight + glyph / 2 - scroll;
		if ( y < - glyph || y > height ) continue;
		const bright = tile === hovered;
		host.label( tile.name.slice( 0, columns ), tile.x, y, bright ? 255 : 200, bright ? 216 : 205, bright ? 102 : 215 );

	}

	renderer.render( scene, camera );

} );
