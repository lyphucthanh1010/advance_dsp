
fs = 44100

def invert(arr):
    map = {}
    for i, a in enumerate(arr):
        map.setdefault(a, []).append(i)
    return map

def chromaprints_comparations(fp_full, fp_frag):
    full_20bit = [x & (1 << 20 - 1) for x in fp_full]
    short_20bit = [x & (1 << 20 - 1) for x in fp_frag]

    common = set(full_20bit) & set(short_20bit)

    i_full_20bit = invert(full_20bit)
    i_short_20bit = invert(short_20bit)

    offsets = {}
    for a in common:
        for i in i_full_20bit[a]:
            for j in i_short_20bit[a]:
                o = i - j
                offsets[o] = offsets.get(o, 0) + 1

    popcnt_table_8bit = [0] * 256
    for i in range(256):
        popcnt_table_8bit[i] = (i & 1) + popcnt_table_8bit[i >> 1]

    def popcnt(x):
        return (popcnt_table_8bit[x & 0xFF] +
                popcnt_table_8bit[(x >> 8) & 0xFF] +
                popcnt_table_8bit[(x >> 16) & 0xFF] +
                popcnt_table_8bit[(x >> 24) & 0xFF])

    def ber(offset):
        errors = 0
        count = 0
        for a, b in zip(fp_full[offset:], fp_frag):
            errors += popcnt(a ^ b)
            count += 1
        return max(0.0, 1.0 - 2.0 * errors / (32.0 * count))

    matches = []
    for count, offset in sorted([(v, k) for k, v in offsets.items()], reverse=True)[:20]:
        matches.append((ber(offset), offset))
    matches.sort(reverse=True)

    score, offset = matches[0]

    offset_secs = int(offset * 0.1238)
    fp_duration = len(fp_frag) * 0.1238 + 2.6476
    return offset_secs, offset_secs + fp_duration, score
