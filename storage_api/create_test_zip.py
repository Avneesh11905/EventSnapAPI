import zipfile
import os

with zipfile.ZipFile('test_images.zip', 'w') as zf:
    # Create fake images
    zf.writestr('image1.jpg', b'fake image data 1')
    zf.writestr('image2.png', b'fake image data 2')
    # Try a fake non-image
    zf.writestr('document.txt', b'this is text, not an image')

print("Created test_images.zip")
