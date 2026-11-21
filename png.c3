// Writes pictures as PNG files (8 bits per channel, RGB, deflate's stored blocks: quick, not
// small), for captures and the rendering tests:
// 
//   png::write("shot.png", pixels, width, height);              // RGBA, sRGB
//   png::write("shot.png", pixels, width, height, p3: true);    // RGBA, in Display P3's colors
// 
// `compress: true` deflates the pixels (smaller, slower to write).
//
// A picture in Display P3 says so in a cICP chunk (P3's primaries and white, sRGB's curve),
// with cHRM and gAMA for readers that predate it; one in sRGB says so in an sRGB chunk.
//
// Chunks of your own go in with `extra` (text in an iTXt, data in a private chunk), and
// find_chunk() gets one back out of a file's bytes:
//
//   png::write("shot.png", pixels, w, h, extra: { { "iTXt", text }, { "abCd", data } });
//   char[] data;
//   if (png::find_chunk(file_bytes, "abCd", &data)) ...
module drie::png;
import std::io, std::collections::list, std::compression::deflate;

uint[256] crc_table @local;
bool crc_ready @local;

fn uint crc32(char[] data, uint crc = 0xffffffff) @local {
    if (!crc_ready) {
        for (uint n = 0; n < 256; n++) {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320 ^ (c >> 1) : c >> 1;
            crc_table[n] = c;
        }
        crc_ready = true;
    }
    foreach (b : data) crc = crc_table[(crc ^ b) & 0xff] ^ (crc >> 8);
    return crc;
}

fn void push_be32(List{char}* out, uint v) @local {
    out.push((char)(v >> 24));
    out.push((char)(v >> 16));
    out.push((char)(v >> 8));
    out.push((char)v);
}

fn void chunk(List{char}* out, String type, char[] data) @local {
    push_be32(out, (uint)data.len);
    sz start = (sz)out.len();
    out.push_all(type);
    out.push_all(data);
    char[] typed = out.entries[start:(sz)out.len() - start];
    push_be32(out, crc32(typed) ^ 0xffffffff);
}

// cHRM's eight numbers: white x, y, then red, green and blue's, times 100000.
fn void chromaticities(List{char}* out, uint[8] xy) @local {
    List{char} data;
    defer data.free();
    foreach (v : xy) push_be32(&data, v);
    chunk(out, "cHRM", data.entries[:(sz)data.len()]);
}

// A chunk to write along with the picture: `type` is its four letters (see the PNG spec for
// what their case means; lowercase-uppercase-uppercase-lowercase, "abCd", is a private chunk
// that readers skip and editors keep).
struct Chunk {
    String type;
    char[] data;
}

// Writes `pixels` (top row first, `channels` bytes each: 4 for RGBA, 3 for RGB; alpha is left
// out) to `path`. `bgra`: blue comes first. `p3`: the colors are Display P3's (the output of a
// pass with DISPLAY_P3_COLOR_SPACE), not sRGB's.
fn bool write(String path, char[] pixels, int width, int height, bool p3 = false, bool bgra = false, int channels = 4,
               Chunk[] extra = {}, bool compress = false) {
    // raw scanlines: filter byte + RGB
    sz row_bytes = (sz)width * 3 + 1;
    sz total = (sz)height * row_bytes;
    char[] raw = mem::alloc_array(char, total);
    defer mem::free(raw);
    sz dst = 0;
    for (int y = 0; y < height; y++) {
        raw[dst++] = 0;
        sz row_src = (sz)y * width * channels;
        if (bgra) {
            for (int x = 0; x < width; x++) {
                sz o = row_src + (sz)x * channels;
                raw[dst++] = pixels[o + 2];
                raw[dst++] = pixels[o + 1];
                raw[dst++] = pixels[o];
            }
        } else {
            for (int x = 0; x < width; x++) {
                sz o = row_src + (sz)x * channels;
                raw[dst++] = pixels[o];
                raw[dst++] = pixels[o + 1];
                raw[dst++] = pixels[o + 2];
            }
        }
    }
    // zlib stream with stored blocks, or deflate's
    List{char} z;
    defer z.free();
    z.push(0x78);
    z.push(compress ? 0x9c : 0x01);
    sz pos = 0;
    if (compress) {
        char[]? packed = deflate::compress(mem, raw);
        if (catch packed) return false;
        defer mem::free(packed);
        z.reserve((usz)packed.len + 6);
        z.push_all(packed);
        // (the stored blocks' loop skipped)
        pos = total;
    }
    while (pos < total || (total == 0 && !compress)) {
        sz n = total - pos;
        if (n > 65535) n = 65535;
        bool last = pos + n >= total;
        z.push(last ? 1 : 0);
        z.push((char)(n & 0xff));
        z.push((char)(n >> 8));
        z.push((char)(~n & 0xff));
        z.push((char)((~n >> 8) & 0xff));
        z.push_all(raw[pos:n]);
        pos += n;
        if (total == 0) break;
    }
    uint a = 1;
    uint b = 0;
    sz ai = 0;
    while (ai < total) {
        sz end = ai + 5552;
        if (end > total) end = total;
        for (; ai < end; ai++) {
            a += raw[ai];
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    push_be32(&z, (b << 16) | a);

    List{char} png;
    png.init(mem, z.len() + 256);
    defer png.free();
    char[8] sig = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    png.push_all(&sig);
    char[13] ihdr;
    ihdr[0] = (char)(width >> 24);
    ihdr[1] = (char)(width >> 16);
    ihdr[2] = (char)(width >> 8);
    ihdr[3] = (char)width;
    ihdr[4] = (char)(height >> 24);
    ihdr[5] = (char)(height >> 16);
    ihdr[6] = (char)(height >> 8);
    ihdr[7] = (char)height;
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // RGB
    chunk(&png, "IHDR", &ihdr);
    if (p3) {
        // colour primaries 12 (P3, D65 white), transfer 13 (sRGB's curve), RGB, full range
        char[4] cicp = { 12, 13, 0, 1 };
        chunk(&png, "cICP", &cicp);
        // (for readers without cICP: P3's chromaticities, and a gamma close to sRGB's curve)
        chromaticities(&png, { 31270, 32900, 68000, 32000, 26500, 69000, 15000, 6000 });
        char[4] gama = { 0, 0, 0xb1, 0x8f }; // 45455: 1 / 2.2
        chunk(&png, "gAMA", &gama);
    } else {
        char[1] intent = { 0 }; // perceptual
        chunk(&png, "sRGB", &intent);
    }
    foreach (c : extra) chunk(&png, c.type, c.data);
    chunk(&png, "IDAT", z.entries[:(sz)z.len()]);
    chunk(&png, "IEND", { });
    if (catch file::save(path, png.entries[:(sz)png.len()])) return false;
    return true;
}

// The data of the first chunk of `type` in a PNG file's bytes (a slice of `file`). False if
// there is none, or the file isn't a PNG.
fn bool find_chunk(char[] file, String type, char[]* data) {
    if (file.len < 8 || file[0] != 0x89 || file[1] != 'P' || file[2] != 'N' || file[3] != 'G') return false;
    usz at = 8;
    while (at + 12 <= file.len) {
        usz len = (usz)file[at] << 24 | (usz)file[at + 1] << 16 | (usz)file[at + 2] << 8 | (usz)file[at + 3];
        if (at + 12 + len > file.len) return false;
        if ((String)file[at + 4:4] == type) {
            *data = file[at + 8:len];
            return true;
        }
        at += 12 + len;
    }
    return false;
}
