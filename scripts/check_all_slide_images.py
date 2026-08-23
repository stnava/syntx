import os
import scripts.generate_presentation as gp

for idx, data in enumerate(gp.SLIDES_DATA):
    img = data.get('image', '')
    exists = os.path.exists(img)
    print(f"Slide {data['num']:02d}: {img} -> Exists: {exists}")
    if not exists:
        print(f"  WARNING: {img} does not exist!")
