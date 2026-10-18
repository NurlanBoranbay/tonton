import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
m = SentenceTransformer("/model", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
                        config_kwargs={"audio_config": None})
imgs = {"14-13-55": "/data/photo_2026-10-17_14-13-55.jpg", "14-14-36": "/data/photo_2026-10-17_14-14-36.jpg"}
I = m.encode([Image.open(p).convert("RGB") for p in imgs.values()], normalize_embeddings=True)
sets = {
 "en short": ("a closed glass door", "an open glass door"),
 "en detailed": ("security camera photo: the glass door is shut, flush with its frame, no gap",
                 "security camera photo: the glass door is swung open, angled away from its frame, gap visible"),
 "ru": ("стеклянная дверь закрыта", "стеклянная дверь открыта"),
}
for name, (closed, opened) in sets.items():
    for prompt in (None, "Classification"):
        T = m.encode([closed, opened], prompt_name=prompt, normalize_embeddings=True)
        S = (I @ T.T)
        out = []
        for k, (sc, so) in zip(imgs, S.tolist()):
            out.append(f"{k}: closed {sc:.3f} open {so:.3f} -> {'OPEN' if so > sc else 'closed'} (margin {so-sc:+.3f})")
        print(f"[{name:11} prompt={prompt}]  " + " | ".join(out))
print("image-image cosine:", float(I[0] @ I[1]))
