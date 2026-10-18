"""Build copy-paste synthetic YOLO set for class 9 (ID-identification).

Document cut-outs (MIDV-2020 photos with doc_quad masks, MIDV-2020 flat templates, synthetic credit
cards) are shrunk to 11-34 px (long side), rotated, colour/blur-degraded and pasted onto real frames of
our cash-desk videos near the places where IDs / money appear in that video. The frame's own labels
are kept. Frames of the v1 val videos are never used as backgrounds.

Outputs (in OUT): images/*.jpg, labels/*.txt (loose boxes like our annotator draws, ~1.3-1.9x card
long side), labels_tight/*.txt (tight box of the pasted card), manifest.csv, stats.txt.
Run: python3 build_yolo_id.py  (CPU only, numpy + Pillow)
"""
import csv, glob, json, math, os, random, re, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

EXT = '/nlp/nurlan/tonton/external'
YOLO = '/nlp/nurlan/tonton/dataset/yolo'
OUT = os.environ.get('OUT', f'{EXT}/yolo_id')
N_IMAGES = int(os.environ.get('N_IMAGES', 3500))
VAL_VIDEOS = {'X-1', 'X-2', 'X-16', 'X-19', 'X-20', 'X-25', 'X-29', 'X-43', 'X-48'}  # v1 val (Latin/Cyrillic X folded)
CUT_MAX = 256  # cut-outs cached at this long side
rng = random.Random(17)


def norm_video(stem):
    return re.sub(r'_f\d+$', '', stem).replace('Х', 'X')  # Cyrillic Х -> Latin X


# ---------------------------------------------------------------- cut-outs
def shrink(im, m=CUT_MAX):
    s = m / max(im.size)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS) if s < 1 else im


def midv_photo_cutouts():
    out = []
    for js in sorted(glob.glob(f'{EXT}/midv2020/photo/annotations/*.json')):
        code = os.path.basename(js)[:-5]
        meta = json.load(open(js))['_via_img_metadata']
        for v in meta.values():
            p = f'{EXT}/midv2020/photo/images/{code}/{v["filename"]}'
            if not os.path.exists(p):
                continue
            quad = [r['shape_attributes'] for r in v['regions'] if r['region_attributes'].get('field_name') == 'doc_quad']
            if not quad:
                continue
            xs, ys = quad[0]['all_points_x'], quad[0]['all_points_y']
            im = Image.open(p).convert('RGB')
            x0, y0, x1, y1 = max(0, min(xs)), max(0, min(ys)), min(im.width, max(xs)), min(im.height, max(ys))
            if x1 - x0 < 50 or y1 - y0 < 50:
                continue
            crop = im.crop((x0, y0, x1, y1))
            mask = Image.new('L', crop.size, 0)
            ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in zip(xs, ys)], fill=255)
            crop.putalpha(mask)
            out.append(('midv_photo', f'{code}/{v["filename"]}', shrink(crop)))
    return out


def midv_template_cutouts():
    out = []
    for p in sorted(glob.glob(f'{EXT}/midv2020/templates/images/*/*.jpg')):
        im = shrink(Image.open(p).convert('RGB'))
        im.putalpha(255)
        out.append(('midv_template', '/'.join(p.split('/')[-2:]), im))
    return out


def creditcard_cutouts():
    out = []
    for p in sorted(glob.glob(f'{EXT}/creditcard_synth/images/train/*.jpg')):
        lp = p.replace('/images/', '/labels/')[:-4] + '.txt'
        if not os.path.exists(lp):
            continue
        im = Image.open(p).convert('RGB')
        for l in open(lp):
            c, x, y, w, h = l.split()
            if c not in ('0', '1'):  # card front / back face
                continue
            x, y, w, h = float(x) * im.width, float(y) * im.height, float(w) * im.width, float(h) * im.height
            crop = shrink(im.crop((round(x - w / 2), round(y - h / 2), round(x + w / 2), round(y + h / 2))))
            mask = Image.new('L', crop.size, 0)
            r = int(0.06 * max(crop.size))
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, crop.width - 1, crop.height - 1], r, fill=255)
            crop.putalpha(mask)
            out.append(('creditcard', os.path.basename(p), crop))
    return out


def perspective(im, strength):
    """Random mild perspective warp of an RGBA image (for flat templates/cards)."""
    w, h = im.size
    d = lambda s: rng.uniform(-strength, strength) * s
    dst = [(d(w), d(h)), (w + d(w), d(h)), (w + d(w), h + d(h)), (d(w), h + d(h))]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    # solve coefficients mapping output (dst) -> input (src)
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        B += [u, v]
    coef = np.linalg.solve(np.array(A, float), np.array(B, float))
    xs, ys = [p[0] for p in dst], [p[1] for p in dst]
    ox, oy = min(xs), min(ys)
    W, H = int(math.ceil(max(xs) - ox)), int(math.ceil(max(ys) - oy))
    # shift output frame by (ox, oy)
    a, b, c, dd, e, f, g, hh = coef
    c2 = a * ox + b * oy + c; f2 = dd * ox + e * oy + f; den = g * ox + hh * oy + 1
    coef2 = [a / den, b / den, c2 / den, dd / den, e / den, f2 / den, g / den, hh / den]
    return im.transform((W, H), Image.PERSPECTIVE, coef2, Image.BICUBIC)


# ---------------------------------------------------------------- backgrounds
def load_labels(stem):
    p = f'{YOLO}/labels/{stem}.txt'
    return [l.split() for l in open(p) if l.strip()] if os.path.exists(p) else []


def main():
    os.makedirs(f'{OUT}/images', exist_ok=True)
    os.makedirs(f'{OUT}/labels', exist_ok=True)
    os.makedirs(f'{OUT}/labels_tight', exist_ok=True)
    cuts = {'midv_photo': midv_photo_cutouts(), 'midv_template': midv_template_cutouts(), 'creditcard': creditcard_cutouts()}
    for k, v in cuts.items():
        print(k, len(v), file=sys.stderr)
    src_w = {'midv_photo': 0.5, 'midv_template': 0.35, 'creditcard': 0.15}
    src_w = {k: v for k, v in src_w.items() if cuts[k]}

    stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{YOLO}/images/*.jpg'))
    stems = [s for s in stems if norm_video(s) not in VAL_VIDEOS and os.path.exists(f'{YOLO}/labels/{s}.txt')]
    anchors_id, anchors_money = defaultdict(list), defaultdict(list)
    for s in stems:
        for c, x, y, w, h in load_labels(s):
            if c == '9':
                anchors_id[norm_video(s)].append((float(x), float(y)))
            elif c == '0':
                anchors_money[norm_video(s)].append((float(x), float(y)))
    global_id = [a for v in anchors_id.values() for a in v]
    print('bg frames', len(stems), 'videos', len({norm_video(s) for s in stems}), file=sys.stderr)

    man = open(f'{OUT}/manifest.csv', 'w', newline='')
    mw = csv.writer(man)
    mw.writerow(['image', 'bg_frame', 'bg_video', 'n_pasted', 'sources', 'doc_long_px'])
    stats = Counter(); longs = []
    for i in range(N_IMAGES):
        s = rng.choice(stems)
        vid = norm_video(s)
        frame = Image.open(f'{YOLO}/images/{s}.jpg').convert('RGB')
        W, H = frame.size
        labels = load_labels(s)
        boxes_id = [(float(x), float(y), float(w), float(h)) for c, x, y, w, h in labels if c == '9']
        new_loose, new_tight, srcs, lp = [], [], [], []
        n = 1 if rng.random() < 0.8 else 2
        for _ in range(n):
            kind = rng.choices(list(src_w), weights=list(src_w.values()))[0]
            name, cut = rng.choice(cuts[kind])[1:]
            if kind != 'midv_photo':
                cut = perspective(cut, 0.12)
            L = math.exp(rng.uniform(math.log(11), math.log(34)))  # doc long side, px
            sc = L / max(cut.size)
            cut = cut.resize((max(2, round(cut.width * sc)), max(2, round(cut.height * sc))), Image.LANCZOS)
            cut = cut.rotate(rng.uniform(0, 360), Image.BICUBIC, expand=True)
            rgb, a = cut.convert('RGB'), cut.getchannel('A')
            # photometric: video looks washed out, low contrast, soft
            rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.85, 1.35))
            rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.55, 1.0))
            rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.4, 1.0))
            a = a.filter(ImageFilter.GaussianBlur(0.3))
            # placement
            r = rng.random()
            pool = anchors_id.get(vid) if r < 0.6 else anchors_money.get(vid) if r < 0.9 else None
            if not pool:
                pool = global_id if r < 0.95 else None
            for _try in range(20):
                if pool:
                    ax, ay = rng.choice(pool)
                    cx, cy = ax * W + rng.gauss(0, 12), ay * H + rng.gauss(0, 8)
                else:
                    cx, cy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.42, 0.65) * H
                x0, y0 = round(cx - cut.width / 2), round(cy - cut.height / 2)
                if x0 < 0 or y0 < 0 or x0 + cut.width > W or y0 + cut.height > H:
                    continue
                aa = np.array(a) > 127
                ys_, xs_ = np.nonzero(aa)
                if len(xs_) == 0:
                    break
                tx0, tx1, ty0, ty1 = x0 + xs_.min(), x0 + xs_.max() + 1, y0 + ys_.min(), y0 + ys_.max() + 1
                tcx, tcy, tw, th = (tx0 + tx1) / 2 / W, (ty0 + ty1) / 2 / H, (tx1 - tx0) / W, (ty1 - ty0) / H
                clash = any(abs(tcx - bx) < (tw + bw) / 2 and abs(tcy - by) < (th + bh) / 2 for bx, by, bw, bh in boxes_id)
                if clash:
                    continue
                frame.paste(rgb, (x0, y0), a)
                # soften the patch like the video codec does
                pad = 3
                bx0, by0, bx1, by1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x0 + cut.width + pad), min(H, y0 + cut.height + pad)
                patch = frame.crop((bx0, by0, bx1, by1))
                f = rng.uniform(1.0, 1.6)
                patch = patch.resize((max(1, round(patch.width / f)), max(1, round(patch.height / f))), Image.BILINEAR).resize(patch.size, Image.BILINEAR)
                frame.paste(patch, (bx0, by0))
                # loose box in the style of our annotator (square-ish, ~1.3-1.9x the card's long side, <=46 px)
                side = min(max(tx1 - tx0, ty1 - ty0) * rng.uniform(1.3, 1.9), 46)  # our boxes: p95 ~35 px
                lw, lh = side * rng.uniform(0.85, 1.15), side * rng.uniform(0.85, 1.15)
                lw, lh = max(lw, tx1 - tx0 + 2), max(lh, ty1 - ty0 + 2)
                lcx, lcy = min(max((tx0 + tx1) / 2, lw / 2), W - lw / 2), min(max((ty0 + ty1) / 2, lh / 2), H - lh / 2)
                new_loose.append(f'9 {lcx / W:.6f} {lcy / H:.6f} {lw / W:.6f} {lh / H:.6f}')
                new_tight.append(f'9 {tcx:.6f} {tcy:.6f} {tw:.6f} {th:.6f}')
                boxes_id.append((lcx / W, lcy / H, lw / W, lh / H))
                srcs.append(f'{kind}:{name}'); lp.append(round(L, 1)); stats[kind] += 1; longs.append(L)
                break
        if not new_loose:
            continue
        name = f'extid_{vid}_{i:05d}'
        frame.save(f'{OUT}/images/{name}.jpg', quality=rng.randint(70, 90))
        keep = [' '.join(l) for l in labels]
        open(f'{OUT}/labels/{name}.txt', 'w').write('\n'.join(keep + new_loose) + '\n')
        open(f'{OUT}/labels_tight/{name}.txt', 'w').write('\n'.join(keep + new_tight) + '\n')
        mw.writerow([name, s, vid, len(new_loose), '|'.join(srcs), '|'.join(map(str, lp))])
        stats['images'] += 1; stats['pasted'] += len(new_loose); stats['kept_bg_boxes'] += len(keep)
        stats['bg_with_real_id'] += any(l[0] == '9' for l in labels)
    man.close()
    q = np.percentile(longs, [5, 25, 50, 75, 95]).round(1)
    txt = [f'built by scripts/build_yolo_id.py, seed 17, N_IMAGES={N_IMAGES}',
           f'images: {stats["images"]}', f'pasted documents (class 9): {stats["pasted"]}',
           f'  from midv_photo: {stats["midv_photo"]}, midv_template: {stats["midv_template"]}, creditcard: {stats["creditcard"]}',
           f'cut-out pools: ' + ', '.join(f'{k}={len(v)}' for k, v in cuts.items()),
           f'kept original frame boxes (all classes): {stats["kept_bg_boxes"]}; frames already holding a real ID: {stats["bg_with_real_id"]}',
           f'pasted doc long side px p5/25/50/75/95: {list(q)}',
           f'background frames: {len(stems)} frames from {len({norm_video(s) for s in stems})} videos (v1 val videos excluded: {sorted(VAL_VIDEOS)})',
           'labels/ = loose square-ish boxes like our annotator; labels_tight/ = tight card box. Other classes copied unchanged.',
           'For CV folds: filter by bg_video in manifest.csv (file name also carries the video).']
    open(f'{OUT}/stats.txt', 'w').write('\n'.join(txt) + '\n')
    print('\n'.join(txt))


if __name__ == '__main__':
    main()
