from PIL import Image, ImageDraw

# Create a simple gradient image
img = Image.new('L', (100, 100), color=255)
draw = ImageDraw.Draw(img)
for i in range(100):
    draw.line([(i, 0), (i, 100)], fill=i * 2)

img.save('test_input.jpg')
print("Created test_input.jpg")
