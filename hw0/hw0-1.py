import sys
Diss = {};
def main():
    from PIL import Image
    sourceFileName = sys.argv[1]
    avatar = Image.open(sourceFileName)
    img = avatar.load()
    # pixels = avatar.load()
    width, height = avatar.size
    for y in range(height):
        for x in range(width):
            rgba = img[x,y]
            rgba = (int(rgba[0]/2),
                    int(rgba[1]/2),
                    int(rgba[2]/2),
                    );
            img[x,y] = rgba
            # avatar.putpixel((x,y), rgba)
    avatar.show()
    avatar.save('Q2.jpg')

    #avatar.save("Q2.jpg")

if __name__ == '__main__':
    main()
