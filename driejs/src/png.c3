// Minimal PNG writer (uncompressed deflate blocks) for offscreen captures.
module main;
import std::io, std::collections::list;

uint[256] crc_table;
bool crc_ready;

fn uint crc32(char[] data, uint crc = 0xffffffff)
{
    if (!crc_ready)
    {
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320 ^ (c >> 1) : c >> 1;
            crc_table[n] = c;
        }
        crc_ready = true;
    }
    foreach (b : data) crc = crc_table[(crc ^ b) & 0xff] ^ (crc >> 8);
    return crc;
}

fn void push_be32(List{char}* out, uint v)
{
    out.push((char)(v >> 24));
    out.push((char)(v >> 16));
    out.push((char)(v >> 8));
    out.push((char)v);
}

fn void png_chunk(List{char}* out, String type, char[] data)
{
    push_be32(out, (uint)data.len);
    sz start = (sz)out.len();
    foreach (c : type) out.push(c);
    foreach (c : data) out.push(c);
    char[] typed = out.entries[start:(sz)out.len() - start];
    push_be32(out, crc32(typed) ^ 0xffffffff);
}

// Writes RGBA pixels (top row first) as an RGB PNG. `bgra` swaps red and blue.
fn bool write_png(String path, char[] rgba, int width, int height, bool bgra = false)
{
    // raw scanlines: filter byte + RGB
    List{char} raw;
    defer raw.free();
    for (int y = 0; y < height; y++)
    {
        raw.push(0);
        for (int x = 0; x < width; x++)
        {
            sz o = ((sz)y * width + x) * 4;
            raw.push(bgra ? rgba[o + 2] : rgba[o]);
            raw.push(rgba[o + 1]);
            raw.push(bgra ? rgba[o] : rgba[o + 2]);
        }
    }
    // zlib stream with stored blocks
    List{char} z;
    defer z.free();
    z.push(0x78);
    z.push(0x01);
    sz total = (sz)raw.len();
    sz pos = 0;
    while (pos < total || total == 0)
    {
        sz n = total - pos;
        if (n > 65535) n = 65535;
        bool last = pos + n >= total;
        z.push(last ? 1 : 0);
        z.push((char)(n & 0xff));
        z.push((char)(n >> 8));
        z.push((char)(~n & 0xff));
        z.push((char)((~n >> 8) & 0xff));
        for (sz i = 0; i < n; i++) z.push(raw[pos + i]);
        pos += n;
        if (total == 0) break;
    }
    uint a = 1;
    uint b = 0;
    for (sz i = 0; i < total; i++)
    {
        a = (a + raw[i]) % 65521;
        b = (b + a) % 65521;
    }
    push_be32(&z, (b << 16) | a);

    List{char} png;
    defer png.free();
    char[8] sig = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    foreach (c : sig) png.push(c);
    char[13] ihdr;
    ihdr[0] = (char)(width >> 24);
    ihdr[1] = (char)(width >> 16);
    ihdr[2] = (char)(width >> 8);
    ihdr[3] = (char)width;
    ihdr[4] = (char)(height >> 24);
    ihdr[5] = (char)(height >> 16);
    ihdr[6] = (char)(height >> 8);
    ihdr[7] = (char)height;
    ihdr[8] = 8;  // bit depth
    ihdr[9] = 2;  // RGB
    png_chunk(&png, "IHDR", &ihdr);
    png_chunk(&png, "IDAT", z.entries[:(sz)z.len()]);
    png_chunk(&png, "IEND", {});
    if (catch file::save(path, png.entries[:(sz)png.len()])) return false;
    return true;
}
