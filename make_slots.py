"""Films v13: the films become NEW Image Control entries; every stock mode stays as it is (FINDINGS §23, §24).

Built from the stock file. New internal tones 25.. (23/24 are the display codes of Custom 1/2), one per film:

  * menu list: count 12 -> 12+N (still and RAW development; movie untouched), one shared index -> tone table
    [stock 12, 25..], tone -> index searches up to 12+N
  * names: tone -> string id 0x380+k (0x532922B4, display-code path 0x532A9860); the two string getters accept ids up
    to 0x3FF and serve 0x380.. from entries in free space (same text in every language)
  * icons: tone -> own icon id per film (0x532935DC, display-code path 0x532A998C; ids of HDF / GR IIIx icons,
    see ICON_IDS), drawn as the Positive Film film strip with a label; display code of a film = its tone
    (GetStillDispImageControl 0x5326C1D0)
  * adjustments: a film uses the settings of PROXY (Standard): the 102 `sub rX,rX,#0xb` tone switches of the 18
    Set*/GetDefaultValue* functions call a stub that maps tones >= 25 to PROXY, and the 27 "is this the active mode"
    checks of PROXY's fields (sets No0/No1/No2) are rewritten in place to also accept tones >= 25, so "apply all"
    (0x532750FC) pushes PROXY's values while a film is selected
  * pipeline: the per-tone parameter pass-through (0x536DB7DC) treats a film as PROXY (otherwise every adjustment
    is forced to neutral); record slot of a film = an unused legacy slot (0x536DC280 -> stub), whose record becomes a copy of the
    Standard record with the film's look matrix; curves as 33-knot tables (film_cave.expand) in library slots 9-11 with
    curve-set selector 4; post matrix through the CSP (film_cave.csp_src). Cross Process gets its stock curves and
    preset chain back on every call.

Free space used (no section grows): the Cross "random" case 0x536EF618.. (areas A, B, D of film_cave/§22o), the
Box-Muller helper 0x536EF088..0x536EF140 (only called from that case), library curves 0-2 (3 x 512 B ROM, reachable
only from that case / saved sets).

Usage: make_slots.py [--raw-nr-above ISO] out.bin name=fit.json [name=fit.json ...]   (order = menu order)
"""
import sys, json, struct
import numpy as np
from gr3patch import Image, RECORDS
from make_film import q13_matrix, SHARED_REC, REC_LEN
from film_cave import csp_src, getter_src, knots_of, expand, csp_entry, N_KNOTS, KNOT_BYTES, CURVE_LIB, STEP
from armasm import assemble

STOCK = '../fwdc239b.bin'
FIRST_TONE = 25
PROXY = 11                                  # Standard: adjustments shared with the films (v13c; v13/v13b used 17 = Positive Film)
LEGACY_SLOTS = (1, 2, 3, 4, 5, 7, 9)        # unreachable records with their own (single-referenced) look matrix
SID0 = 0x380                                # string ids of the film names
# icon ids taken over for the films: HDF on/off (0x05-0x07, HDF models only), GR IIIx certification marks (0x1D,
# 720x360; the GR III shows 0x1E) and GR IIIx TELE / 71mm crop (0x2D, 0x2E). 0x05-0x07 are 40x40 and are redrawn in
# place; 0x1D/0x2D/0x2E get 40x40 descriptors inside 0x1D's pixel area. Only for a GR III without HDF.
ICON_IDS = (0x05, 0x06, 0x07, 0x1D, 0x2D, 0x2E)
CARVE_FROM = 0x1D
FRAME_ICON = 0xE8                           # Positive Film icon: film-strip frame for the labels
ICON_LABELS = dict(                         # key -> (label, colour), as make_icons LABELS_V10/V12
    portra=('P400', (255, 214, 64)), ektar=('E100', (255, 120, 80)), gold=('G200', (255, 170, 30)),
    pro400h=('400H', (120, 220, 140)), vision3=('250D', (90, 190, 255)), classicneg=('NC', (110, 205, 200)),
    ultramax=('UM400', (255, 200, 120)), xtra=('X400', (120, 200, 120)), vision350d=('50D', (150, 210, 255)))
SID_MAX = 0x3FF

B = lambda at, to, cond=0xE: (cond << 28) | 0x0A000000 | (((to - at - 8) >> 2) & 0xFFFFFF)
BL = lambda at, to: 0xEB000000 | (((to - at - 8) >> 2) & 0xFFFFFF)
NOP = 0xE1A00000

# free regions (start, end)
RANDOM, PRESET = 0x536EF618, 0x536EF6DC
REGIONS = [(0x536EF61C, 0x536EF6DC), (0x536EF6FC, 0x536EF998), (0x536EF088, 0x536EF140), (0x536EF99C, 0x536EF9B8)]
CURVE_SLOTS_FREE = (0, 1, 2)                # library curves whose ROM storage is reused

# list / UI
COUNT_AT = 0x532C2ACC                       # movls r0,#0xc (still)
IDX2TONE_PAIRS = [(0x532C2D60, 0x532C2D64, 0x53B59224), (0x532C2D80, 0x532C2D84, 0x53B58FCC),
                  (0x532C2D00, 0x532C2D04, 0x53B5919C)]
TONE2IDX_PAIRS = [(0x532C2F58, 0x532C2F60, 0x53B5919C), (0x532C2FF0, 0x532C2FF8, 0x53B58FCC),
                  (0x532C3074, 0x532C307C, 0x53B59224)]
TONE2IDX_CMP = (0x532C2F74, 0x532C300C, 0x532C3090)
NAME_FN, ICON_FN = 0x532922B4, 0x532935DC  # first word cmp r1,#0xa
DISP_DEFAULT, DISP_RET, DISP_FAIL = 0x5326C240, 0x5326C22C, 0x5326C2A0
DSTR_DEFAULT, DSTR_RET, DSTR_FAIL = 0x532A9874, 0x532A98E0, 0x532A9958
DICON_DEFAULT, DICON_RET, DICON_FAIL = 0x532A99A0, 0x532A9A0C, 0x532A9A84
STR1_BOUND, STR1_LOAD = 0x533966D4, 0x533966FC     # movw r3,#0x311 ; ldr r7,[r3,r7,lsl #2]
STR2_BOUND, STR2_LOAD = 0x533970EC, 0x53397108     # movw r3,#0x311 ; ldr r3,[r3,r7,lsl #2]
PROXY_FUNCS = (0x532BD784, 0x532BDCD8, 0x532BE22C, 0x532BE8A0, 0x532BEF14, 0x532BF588, 0x532BFBFC, 0x532C0DA8,
               0x532C141C, 0x532C34E8, 0x532C3B54, 0x532C41C0, 0x532C494C, 0x532C50D8, 0x532C5864, 0x532C5FF0,
               0x532C8BD4, 0x532C9360)      # Set*/GetDefaultValue* x {Sat, Hue, Key, Contrast, HL, Shadow, Sharp, Shading, Clarity}
# pipeline
SLOT_AT = 0x536DC280                        # cmp r2,#0x16 ; movne r1,r2 ; moveq r1,#0x19
PARAM_AT = 0x536DB7FC                       # cmp r2,#0xb: per-tone parameter pass-through 0x536DB7DC (other tones
                                            # get every adjustment forced to neutral)
FILL_LDR, FILL_AT = 0x536DD450, 0x536DD454  # ldrne r3,[r6,#0x2d0] ; beq 0x536DD4B8 (Cross preset chain)
CHAIN, NO_CURVES = 0x536DD4B8, 0x536DD458
CSP_CALL, CSP_CALC = 0x536E4774, 0x536E4720
GET_CALL, GETTER = 0x536E87C4, 0x536EEDF0
RET_WORD = 0xE89DAFF8                       # ldm sp,{r3-r10,fp,sp,pc} (epilogue of the ctx fill)


def movw(rd, v): return 0xE3000000 | (rd << 12) | ((v & 0xF000) << 4) | (v & 0xFFF)
def movt(rd, v): return 0xE3400000 | (rd << 12) | ((v >> 16 & 0xF000) << 4) | (v >> 16 & 0xFFF)


class Space:
    """First-fit allocator over the free regions; code is assembled at its final address."""
    def __init__(self, regions):
        self.free = [list(r) for r in regions]
        self.used = []

    def take(self, n, name, prefer=None):
        reserved = (n + 3) & ~3
        order = sorted(range(len(self.free)), key=lambda i: (prefer is not None and i not in prefer, i))
        for i in order:
            lo, hi = self.free[i]
            if hi - lo >= reserved:
                self.free[i][0] = lo + reserved
                self.used.append((name, lo, n))
                return lo
        raise SystemExit(f'no room for {name} ({n} B); free: {[(hex(a), b - a) for a, b in self.free]}')


def code(space, img, name, src_fn, prefer=None):
    n = len(assemble(src_fn(0x53000000), 0x53000000))
    at = space.take(4 * n, name, prefer)
    words = assemble(src_fn(at), at)
    assert len(words) == n
    img.write(at, struct.pack(f'<{n}I', *words))
    return at


def put(img, at, word, expect):
    got = img.word(at)
    assert got == expect, f'{at:#x}: {got:#010x} != {expect:#010x}'
    img.write(at, struct.pack('<I', word))


def expand_multi_src(ranges):
    """film_cave.expand_src with several knot ranges [(lo, n_bytes)]."""
    tests = '\n'.join(f'''        movw r2, #{lo & 0xFFFF:#x}
        movt r2, #{lo >> 16:#x}
        sub r3, r1, r2
        cmp r3, #{n}
        blo X''' for lo, n in ranges)
    return f'''
        ldr r1, [r0]
{tests}
        bx lr
    X:  str r7, [r0]
        mov r2, #0
    E:  lsr r3, r2, #3
        add r3, r1, r3, lsl #1
        ldrh r12, [r3]
        ldrh r3, [r3, #2]
        sub r3, r3, r12
        and r8, r2, #7
        mul r3, r8, r3
        add r12, r12, r3, asr #3
        strh r12, [r7], #2
        add r2, r2, #1
        cmp r2, #256
        bne E
        bx lr
    '''


def patch(img, films, raw_nr_above=None):
    """films: [(name, display name, fit)] in menu order."""
    n = len(films)
    assert 1 <= n <= len(LEGACY_SLOTS)
    space = Space(REGIONS)
    lib = [img.word(CURVE_LIB + 4 * i) for i in range(12)]
    kspace = Space([(lib[i], lib[i] + 512) for i in CURVE_SLOTS_FREE])
    info = {}

    # --- film data: records, matrices, knots, CSP table --------------------------------------------------------
    shared = bytes(img.read(SHARED_REC, REC_LEN))
    slots, knot_ptrs, posts, knot_ranges = [], [], {}, []
    for k, (key, label, fit) in enumerate(films):
        slot = LEGACY_SLOTS[k]
        rec = img.word(RECORDS + 4 * slot)
        mat = img.word(rec + 4)
        data = bytearray(shared)
        data[4:8] = struct.pack('<I', mat)
        img.write(rec, bytes(data))
        q = q13_matrix(fit['matrix'])
        img.write(mat, struct.pack('<9h', *q.ravel().tolist()))
        kn = [knots_of(c) for c in fit['presets'][0]['curves']]
        err = max(int(abs(expand(x) - np.asarray(c)).max()) for x, c in zip(kn, fit['presets'][0]['curves']))
        at = kspace.take(KNOT_BYTES, f'knots {key}')
        img.write(at, struct.pack(f'<{3 * N_KNOTS}H', *np.ravel(kn).tolist()))
        knot_ranges.append((at, KNOT_BYTES))
        knot_ptrs += [at + 2 * N_KNOTS * c for c in range(3)]
        P = fit.get('post_q512', fit.get('cca1_q512'))
        if P is not None:
            posts[rec] = P
        slots.append(slot)
        info[key] = dict(tone=FIRST_TONE + k, name=label, slot=slot, record=hex(rec), matrix=hex(mat),
                         knots=hex(at), knot_err=err, post=P)
    # neighbouring knot ranges merged for the expander
    knot_ranges.sort()
    merged = []
    for lo, m in knot_ranges:
        if merged and merged[-1][0] + merged[-1][1] == lo:
            merged[-1][1] += m
        else:
            merged.append([lo, m])
    for lo, m in merged:
        assert m < 256 or m % 4 == 0

    def data(blob, name, prefer=None, sp=None):
        sp = sp or kspace
        at = sp.take(len(blob), name) if prefer is None else sp.take(len(blob), name, prefer)
        img.write(at, blob)
        return at

    # names: entries {u8 chars incl. NUL, u8 1, u16 0, u32 -> UTF-16LE}
    texts = [data((label + '\0').encode('utf-16le'), f'text {key}') for key, label, _ in films]
    entries = data(b''.join(struct.pack('<BBHI', len(label) + 1, 1, 0, t) for (_, label, _), t in zip(films, texts)),
                   'string entries')
    order = bytes(img.read(0x53B59224, 12)) + bytes(range(FIRST_TONE, FIRST_TONE + n))
    idx_table = data(order, 'index->tone table')
    slot_table = data(bytes(slots), 'tone->slot table')
    ptr_table = data(struct.pack(f'<{3 * n}I', *knot_ptrs), 'knot pointer table')
    stock6 = data(struct.pack('<6I', *lib[6:12]), 'stock library pointers 6-11')
    csp_table = b''.join(csp_entry(r, P) for r, P in posts.items()) + csp_entry(0, [[512, 0, 0], [0, 512, 0], [0, 0, 512]])
    csp_tab_at = data(csp_table, 'CSP table', sp=space)

    # --- pipeline code -----------------------------------------------------------------------------------------
    t0 = FIRST_TONE
    slot_at = code(space, img, 'slot map', lambda at: f'''
        cmp r2, #0x16
        moveq r1, #0x19
        bxeq lr
        sub r1, r2, #{t0}
        cmp r1, #{n}
        movhs r1, r2
        bxhs lr
        movw ip, #{slot_table & 0xFFFF:#x}
        movt ip, #{slot_table >> 16:#x}
        ldrb r1, [ip, r1]
        bx lr''')
    param_at = code(space, img, 'param tone', lambda at: f'''
        cmp r2, #{t0}
        movhs r2, #{PROXY}
        cmp r2, #0xb
        bx lr''')
    lib6 = CURVE_LIB + 24
    fill_at = code(space, img, 'curve fill', lambda at: f'''
        movw r0, #{lib6 & 0xFFFF:#x}
        movt r0, #{lib6 >> 16:#x}
        movw r2, #{stock6 & 0xFFFF:#x}
        movt r2, #{stock6 >> 16:#x}
        ldm r2, {{r4, r5, r8, r9, r10, r12}}
        stm r0, {{r4, r5, r8, r9, r10, r12}}
        cmp r1, #0x15
        beq abs({CHAIN:#x})
        sub r2, r1, #{t0}
        cmp r2, #{n}
        bhs abs({NO_CURVES:#x})
        add r2, r2, r2, lsl #1
        movw r4, #{ptr_table & 0xFFFF:#x}
        movt r4, #{ptr_table >> 16:#x}
        add r4, r4, r2, lsl #2
        ldm r4, {{r5, r8, r9}}
        add r0, r0, #12
        stm r0, {{r5, r8, r9}}
        add r3, r3, #0x38000
        mov r2, #4
        strb r2, [r3, #0xa98]
        .word {RET_WORD:#x}''', prefer=[0])
    csp_at = code(space, img, 'CSP stub', lambda at: csp_src(csp_tab_at), prefer=[1])
    exp_at = code(space, img, 'knot expander', lambda at: expand_multi_src(merged), prefer=[1])
    get_at = code(space, img, 'curve getter', lambda at: getter_src(exp_at), prefer=[1])

    # --- UI code -----------------------------------------------------------------------------------------------
    stubs = {}
    for r in (4, 5):
        stubs[r] = code(space, img, f'proxy stub r{r}', lambda at, r=r: f'''
        cmp r{r}, #{t0}
        movhs r{r}, #{PROXY - 11}
        sublo r{r}, r{r}, #11
        bx lr''')
    name_at = code(space, img, 'name', lambda at: f'''
        sub ip, r1, #{t0}
        cmp ip, #{n}
        addlo r0, ip, #{SID0:#x}
        bxlo lr
        cmp r1, #0xa
        b abs({NAME_FN + 4:#x})''')
    icon_tab = data(struct.pack(f'<{n}H', *ICON_IDS[:n]), 'icon id table')
    icon_at = code(space, img, 'icon', lambda at: f'''
        sub ip, r1, #{t0}
        cmp ip, #{n}
        bhs F
        movw r0, #{icon_tab & 0xFFFF:#x}
        movt r0, #{icon_tab >> 16:#x}
        add ip, r0, ip, lsl #1
        ldrh r0, [ip]
        bx lr
    F:  cmp r1, #0xa
        b abs({ICON_FN + 4:#x})''')
    disp_at = code(space, img, 'display code', lambda at: f'''
        sub ip, r3, #{t0}
        cmp ip, #{n}
        movlo r0, r3
        blo abs({DISP_RET:#x})
        b abs({DISP_FAIL:#x})''')
    dstr_at = code(space, img, 'display string', lambda at: f'''
        sub ip, r1, #{t0}
        cmp ip, #{n}
        addlo r0, ip, #{SID0:#x}
        blo abs({DSTR_RET:#x})
        b abs({DSTR_FAIL:#x})''')
    dicon_at = code(space, img, 'display icon', lambda at: f'''
        sub ip, r1, #{t0}
        cmp ip, #{n}
        bhs abs({DICON_FAIL:#x})
        movw r0, #{icon_tab & 0xFFFF:#x}
        movt r0, #{icon_tab >> 16:#x}
        add ip, r0, ip, lsl #1
        ldrh r0, [ip]
        b abs({DICON_RET:#x})''')

    def strfix(rd):
        return lambda at: f'''
        subs ip, r7, #{SID0:#x}
        blo L
        cmp ip, #{n}
        movw r{rd}, #{entries & 0xFFFF:#x}
        movt r{rd}, #{entries >> 16:#x}
        addlo r{rd}, r{rd}, ip, lsl #3
        movhs r{rd}, #0
        bx lr
    L:  ldr r{rd}, [r3, r7, lsl #2]
        bx lr'''
    str1_at = code(space, img, 'string getter 1', strfix(7))
    str2_at = code(space, img, 'string getter 2', strfix(3))

    # --- hooks -------------------------------------------------------------------------------------------------
    put(img, RANDOM, B(RANDOM, PRESET), 0xEDDF0BE0)                       # random case -> fixed-preset case
    put(img, SLOT_AT, BL(SLOT_AT, slot_at), 0xE3520016)
    put(img, SLOT_AT + 4, NOP, 0x11A01002)
    put(img, SLOT_AT + 8, NOP, 0x03A01019)
    put(img, PARAM_AT, BL(PARAM_AT, param_at), 0xE352000B)
    put(img, FILL_LDR, 0xE59632D0, 0x159632D0)                           # ldrne -> ldr r3,[r6,#0x2d0]
    put(img, FILL_AT, B(FILL_AT, fill_at), B(FILL_AT, CHAIN, 0x0))
    put(img, CSP_CALL, BL(CSP_CALL, csp_at), BL(CSP_CALL, CSP_CALC))
    put(img, GET_CALL, BL(GET_CALL, get_at), BL(GET_CALL, GETTER))
    # list
    put(img, COUNT_AT, 0x93A00000 | (12 + n), 0x93A0000C)
    for lo_at, hi_at, old in IDX2TONE_PAIRS + TONE2IDX_PAIRS:
        put(img, lo_at, movw(3, idx_table), movw(3, old))
        put(img, hi_at, movt(3, idx_table), movt(3, old))
    for at in TONE2IDX_CMP:
        put(img, at, 0xE3500000 | (12 + n), 0xE350000C)
    # names / icons / display codes
    put(img, NAME_FN, B(NAME_FN, name_at), 0xE351000A)
    put(img, ICON_FN, B(ICON_FN, icon_at), 0xE351000A)
    put(img, DISP_DEFAULT, B(DISP_DEFAULT, disp_at), B(DISP_DEFAULT, DISP_FAIL))
    put(img, DSTR_DEFAULT, B(DSTR_DEFAULT, dstr_at), B(DSTR_DEFAULT, DSTR_FAIL))
    put(img, DICON_DEFAULT, B(DICON_DEFAULT, dicon_at), B(DICON_DEFAULT, DICON_FAIL))
    put(img, STR1_BOUND, movw(3, SID_MAX), movw(3, 0x311))
    put(img, STR1_LOAD, BL(STR1_LOAD, str1_at), 0xE7937107)
    put(img, STR2_BOUND, movw(3, SID_MAX), movw(3, 0x311))
    put(img, STR2_LOAD, BL(STR2_LOAD, str2_at), 0xE7933107)
    # adjustments: tone switches
    n_sw = 0
    for f in PROXY_FUNCS:
        e = f + 4
        while img.word(e) != 0xE1A0C00D:
            e += 4
        for at in range(f, e, 4):
            wd = img.word(at)
            r = (wd >> 12) & 15
            if (wd & 0xFFF00FFF) == 0xE240000B and (wd >> 16) & 15 == r and img.word(at + 4) == (0xE350000B | r << 16) \
                    and (img.word(at + 8) & 0xFFFFF000) == 0x979FF000:
                assert r in stubs, hex(at)
                put(img, at, BL(at, stubs[r]), wd)
                n_sw += 1
    assert n_sw == 102, n_sw
    # adjustments: PROXY's "is active mode" checks
    n_chk = patch_checks(img)
    info['_code'] = dict(n_tone_switches=n_sw, n_checks=n_chk, idx_table=hex(idx_table), entries=hex(entries),
                         space=[(name, hex(at), size) for name, at, size in space.used + kspace.used],
                         free_left=[(hex(a), b - a) for a, b in space.free + kspace.free])
    info['_icons'] = patch_film_icons(img, [key for key, _, _ in films])
    if raw_nr_above:
        from raw_nr import patch_raw_nr_min_iso, above_index
        info['_raw_nr_above'] = patch_raw_nr_min_iso(img, first=above_index(raw_nr_above))
    return info


def patch_film_icons(img, keys):
    """Point ICON_IDS[3:] at 40x40 slices of CARVE_FROM's pixels and draw a labelled film-strip icon per film."""
    from make_icons import descriptor, get_icon, put_icon, render, CATALOG
    frame = get_icon(img, FRAME_ICON)
    assert frame.shape == (40, 40, 4)
    w, h, base = descriptor(img, CARVE_FROM)
    assert (w, h) == (720, 360)
    j = 0
    out = {}
    for key, icon_id in zip(keys, ICON_IDS):
        d = img.word(CATALOG + 4 * icon_id)
        if descriptor(img, icon_id)[:2] != (40, 40):
            img.write(d, struct.pack('<HHI', 40, 40, base + 1600 * j))
            j += 1
        assert descriptor(img, icon_id)[:2] == (40, 40)
        label, colour = ICON_LABELS[key]
        put_icon(img, icon_id, render(frame, label, colour))
        out[key] = (hex(icon_id), label)
    assert j <= w * h // 1600
    return out


CHECK0 = [0xE5903004, 0xE5D32044, 0xE3520000, None, 0xE59331C0, 0xE3C334FF, 0xE3C33CFF, None, None, 0xE3A00000,
          0xE12FFF1E, None]


def check_sites(img, tone):
    """[(address, set, tail)] of the 'is active mode' checks of `tone` in the still user-settings setters."""
    rt = img.rtos()
    W = np.frombuffer(rt[:len(rt) // 4 * 4], '<u4')
    base = 0x53000000
    out = []
    k0 = 0xE3530000 | 0x800 | tone          # cmp r3,#tone<<16 (imm8 ror 16)
    for i in np.flatnonzero(W == 0xE59331C0):
        s = i - 4
        ws = [int(x) for x in W[s:s + 12]]
        if all(p is None or p == w for p, w in zip(CHECK0, ws)) and ws[7] == k0 \
                and ws[3] == B(base + 4 * (s + 3), base + 4 * (s + 9), 0x1) \
                and ws[8] == B(base + 4 * (s + 8), base + 4 * (s + 11), 0x0) and (ws[11] >> 24) == 0xEA:
            tail = base + 4 * (s + 11) + 8 + (((ws[11] & 0xFFFFFF) ^ 0x800000) - 0x800000) * 4
            out.append((base + 4 * s, 0, tail))
    for off, setv in ((0x1C4, 1), (0x1C6, 2)):
        for i in np.flatnonzero(W == (0xE5D33000 | off)):
            s = i - 9
            ws = [int(x) for x in W[s:s + 13]]
            exp = [0xE5903004, 0xE5D32044, 0xE3520000, None, 0xE5D321C0, 0xE3520000 | setv, None, 0xE3A00000,
                   0xE12FFF1E, 0xE5D33000 | off, 0xE3530000 | tone, None, None]
            if all(p is None or p == w for p, w in zip(exp, ws)) \
                    and ws[3] == B(base + 4 * (s + 3), base + 4 * (s + 7), 0x1) \
                    and ws[6] == B(base + 4 * (s + 6), base + 4 * (s + 9), 0x0) \
                    and ws[11] == B(base + 4 * (s + 11), base + 4 * (s + 7), 0x1) and (ws[12] >> 24) == 0xEA:
                tail = base + 4 * (s + 12) + 8 + (((ws[12] & 0xFFFFFF) ^ 0x800000) - 0x800000) * 4
                out.append((base + 4 * s, setv, tail))
    return out


def patch_checks(img):
    sites = check_sites(img, PROXY)
    assert len(sites) == 27, len(sites)
    for at, setv, tail in sites:
        if setv == 0:
            src = f'''
        ldr r3, [r0, #4]
        ldrb r2, [r3, #0x44]
        ldrb ip, [r3, #0x1c0]
        orrs r2, r2, ip
        bne R
        ldrb r3, [r3, #0x1c2]
        cmp r3, #{FIRST_TONE}
        bhs abs({tail:#x})
        cmp r3, #{PROXY}
        beq abs({tail:#x})
    R:  mov r0, #0
        bx lr'''
            size = 12
        else:
            src = f'''
        ldr r3, [r0, #4]
        ldrb r2, [r3, #0x44]
        ldrb ip, [r3, #0x1c0]
        ldrb r3, [r3, #{0x1C2 + 2 * setv:#x}]
        cmp ip, #{setv}
        cmpeq r2, #0
        bne R
        cmp r3, #{FIRST_TONE}
        bhs abs({tail:#x})
        cmp r3, #{PROXY}
        beq abs({tail:#x})
    R:  mov r0, #0
        bx lr'''
            size = 13
        words = assemble(src, at)
        assert len(words) == size
        img.write(at, struct.pack(f'<{size}I', *words))
    return len(sites)


FILMS = dict(  # key -> (menu name)
    portra='Portra 400', ektar='Ektar 100', gold='Gold 200', pro400h='Pro 400H', vision3='Vision3 250D',
    classicneg='Classic Neg', ultramax='UltraMax 400', xtra='Superia X-TRA', vision350d='Vision3 50D')

if __name__ == '__main__':
    args = sys.argv[1:]
    raw_nr_above = None
    if '--raw-nr-above' in args:
        i = args.index('--raw-nr-above')
        raw_nr_above = int(args[i + 1])
        del args[i:i + 2]
    out = args[0]
    films = []
    for a in args[1:]:
        key, path = a.split('=', 1)
        films.append((key, FILMS[key], json.load(open(path))))
    img = Image(STOCK)
    info = patch(img, films, raw_nr_above)
    data, rep = img.build()
    open(out, 'wb').write(data)
    for k, v in info.items():
        print(k, v)
    print(rep)
