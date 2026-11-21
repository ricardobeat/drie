// three's UltraHDRLoader with its gain map applied natively (gainmapApply in src/main.c3):
// three's loop over every channel in JS takes a second for a 2K image. The JPEGs are decoded
// by drie and the gain map stretched to the SDR image's size bilinearly, as the canvas does
// in three's version; parsing and the result are three's.
import { HalfFloatType } from "../vendor/three.core.js";
import { UltraHDRLoader as ThreeUltraHDRLoader } from "../vendor/jsm/loaders/UltraHDRLoader.js";

const N = globalThis.__drie;

class UltraHDRLoader extends ThreeUltraHDRLoader {

	_applyGainmapToSDR( metadata, sdrBuffer, gainmapBuffer, onSuccess, onError ) {

		const sdr = N.imageDecode( sdrBuffer );
		const gain = N.imageDecode( gainmapBuffer );
		try {

			if ( sdr < 0 || gain < 0 ) throw new Error( 'THREE.UltraHDRLoader: Could not decode the JPEGs' );
			const width = N.imageSize( sdr, 0 ), height = N.imageSize( sdr, 1 );
			const gw = N.imageSize( gain, 0 ), gh = N.imageSize( gain, 1 );
			if ( width / height !== gw / gh ) {

				onError( 'THREE.UltraHDRLoader Error: Aspect ratio mismatch between SDR and Gainmap images' );
				return;

			}

			const length = width * height * 4;
			const sdrPixels = new Uint8Array( length ), gainPixels = new Uint8Array( length );
			N.imagePixels( sdr, sdrPixels, width, height );
			N.imagePixels( gain, gainPixels, width, height );

			const maxDisplayBoost = 1.8 ** ( metadata.hdrCapacityMax * 0.5 );
			const weightFactor = Math.min( Math.max(
				( Math.log2( maxDisplayBoost ) - metadata.hdrCapacityMin ) / ( metadata.hdrCapacityMax - metadata.hdrCapacityMin ),
				0.0 ), 1.0 );
			const hdr = this.type === HalfFloatType
				? new Uint16Array( length ).fill( 15360 )
				: new Float32Array( length ).fill( 1.0 );
			N.gainmapApply( sdrPixels, gainPixels, hdr, new Float32Array( [
				metadata.gainMapMin, metadata.gainMapMax, metadata.offsetSDR, metadata.offsetHDR,
				1.0 / metadata.gamma, weightFactor ] ) );
			onSuccess( hdr, width, height );

		} catch ( e ) {

			onError( e );

		} finally {

			if ( sdr >= 0 ) N.imageFree( sdr );
			if ( gain >= 0 ) N.imageFree( gain );

		}

	}

}

export { UltraHDRLoader };
