// The shadow atlas's space, handed out as a quadtree of square blocks: a block is a quarter of
// one twice its size, and four free quarters make their parent free again. A block goes to the
// free place nearest the atlas's corner (in Z order), so the part in use stays packed there and
// the texture only needs to be as large as that part.
module drie;
import std::collections::list;

// The smallest block, in pixels.
const int ATLAS_MIN_BLOCK = 128;
const int ATLAS_LEVELS @local = 9;

struct AtlasPacker @private {
    // the side of the whole quadtree (a power of two, ATLAS_MIN_BLOCK or more)
    int root;
    // the free blocks of each level (0: the root), as atlas_key()s
    List{uint}[ATLAS_LEVELS] free;
}

// A block's place, in ATLAS_MIN_BLOCK steps: x in the high half, y in the low one.
fn uint atlas_key(int x, int y) @private => (uint)(x / ATLAS_MIN_BLOCK) << 16 | (uint)(y / ATLAS_MIN_BLOCK);
fn int atlas_key_x(uint key) @private => (int)(key >> 16) * ATLAS_MIN_BLOCK;
fn int atlas_key_y(uint key) @private => (int)(key & 0xffff) * ATLAS_MIN_BLOCK;

// Z order of a key: its x and y bits interleaved.
fn ulong atlas_z(uint key) @local {
    ulong z;
    for (int b = 0; b < 16; b++) {
        z |= (ulong)((key >> b) & 1) << (2 * b);
        z |= (ulong)((key >> (16 + b)) & 1) << (2 * b + 1);
    }
    return z;
}

// Empties the packer, `root` pixels on a side (rounded down to a power of two and to the levels
// it has).
fn void AtlasPacker.reset(&self, int root) {
    int r = ATLAS_MIN_BLOCK;
    while (r * 2 <= root && r < ATLAS_MIN_BLOCK << (ATLAS_LEVELS - 1)) r *= 2;
    self.root = r;
    foreach (&l : self.free) {
        if (!l.allocator) l.init(mem);
        l.clear();
    }
    self.free[0].push(atlas_key(0, 0));
}

// The level of blocks `size` pixels on a side (-1: larger than the root).
fn int AtlasPacker.level(&self, int size) {
    int level;
    for (int s = self.root; s > size; s /= 2) level++;
    return (self.root >> level) == size && size >= ATLAS_MIN_BLOCK ? level : -1;
}

// A free block of `level`: the one first in Z order, or a quarter of one a level up.
fn bool AtlasPacker.take(&self, int level, uint* out) {
    if (level < 0 || level >= ATLAS_LEVELS) return false;
    List{uint}* l = &self.free[level];
    sz best = -1;
    ulong best_z;
    foreach (i, k : *l) {
        ulong z = atlas_z(k);
        if (best < 0 || z < best_z) {
            best = (sz)i;
            best_z = z;
        }
    }
    if (best >= 0) {
        *out = (*l)[best];
        l.remove_at(best);
        return true;
    }
    uint parent;
    if (!self.take(level - 1, &parent)) return false;
    int half = (self.root >> level);
    int x = atlas_key_x(parent);
    int y = atlas_key_y(parent);
    l.push(atlas_key(x + half, y));
    l.push(atlas_key(x, y + half));
    l.push(atlas_key(x + half, y + half));
    *out = parent;
    return true;
}

// A block `size` pixels on a side (a power of two, ATLAS_MIN_BLOCK or more): false when the
// atlas has no room for it.
fn bool AtlasPacker.alloc(&self, int size, uint* out) => self.take(self.level(size), out);

// Gives back block `key`, `size` pixels on a side.
fn void AtlasPacker.release(&self, uint key, int size) {
    int level = self.level(size);
    if (level < 0) return;
    if (level == 0) {
        self.free[0].push(key);
        return;
    }
    // its three siblings free too: their parent is
    int side = size * 2;
    int px = atlas_key_x(key) / side * side;
    int py = atlas_key_y(key) / side * side;
    uint[4] quarter = { atlas_key(px, py), atlas_key(px + size, py), atlas_key(px, py + size), atlas_key(px + size, py + size) };
    List{uint}* l = &self.free[level];
    int found;
    foreach (q : quarter) {
        if (q == key) continue;
        foreach (k : *l) {
            if (k == q) {
                found++;
                break;
            }
        }
    }
    if (found < 3) {
        l.push(key);
        return;
    }
    for (sz i = l.len(); i > 0; i--) {
        uint k = (*l)[i - 1];
        foreach (q : quarter) {
            if (k == q) {
                l.remove_at(i - 1);
                break;
            }
        }
    }
    self.release(quarter[0], side);
}
