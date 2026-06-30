# Take-home: Spot the Fake Photo

Full brief: **ASSIGNMENT.pdf**. In short:

**Task** — Given one image, decide if it's a **real photo** or a **photo of a screen**
(someone re-photographing a phone/laptop instead of the real thing).

**The bar:** aim for **>95% accuracy** on our held-out photos.

**Do this**
1. Take ~50 real photos + ~50 photos-of-a-screen with your phone → folders `real/` and `screen/`.
2. **Solve it any way you like — training a model is *not* required.** A trained model, classic
   CV / image-processing tricks, frequency analysis, any algorithm — figuring out the approach
   is the real test. Keep it small and fast.
3. Make `python predict.py image.jpg` print a number 0–1 (1 = photo-of-a-screen). A starter
   `predict.py` is here — just fill it in.

**Send us**
- Your code (`predict.py` + training code)
- A short note (½ page): approach, your accuracy, what you'd improve
- **Two numbers (required):** latency (ms per image, on what device) and cost per image
  (on-device ≈ free, or a rough $ per 1,000 / per million images for a cloud server)
- Optional: a tiny camera demo (web page)

**We judge** by running your `predict.py` on our own photos, reading your note, and looking at
your latency + cost-per-image. Small + fast + cheap + honest beats big + complicated.

~1 day. Use whatever tools you like.
