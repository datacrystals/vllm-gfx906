#!/usr/bin/env python3
"""Generate 'a red box moving right on black' test video (256x256, 2s, 24fps)."""
import av, numpy as np

W = H = 256
NF = 48
out = av.open('/data/tmp/omni_assets/test_video2.mp4', 'w')
st = out.add_stream('libx264', rate=24)
st.width, st.height = W, H
st.pix_fmt = 'yuv420p'
st.options = {'crf': '18'}
for i in range(NF):
    a = np.zeros((H, W, 3), dtype=np.uint8)
    x = 20 + int(i * (W - 80) / (NF - 1))
    a[96:160, x:x+40] = (220, 30, 30)
    fr = av.VideoFrame.from_ndarray(a, format='rgb24')
    for p in st.encode(fr):
        out.mux(p)
for p in st.encode(None):
    out.mux(p)
out.close()
print('wrote test_video2.mp4')
