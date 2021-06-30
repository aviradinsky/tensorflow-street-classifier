
import matplotlib.pyplot as plt
import numpy as np


from PIL import Image, ImageDraw as D


#crops the images into 
def movingWindow(image):
    stepSize = 200
    pictures = []
    img = Image.open(image)
    img.show()
    for y in range(0, img.height, stepSize):
        for x in range(0, img.width, stepSize):
            newx = x+stepSize
            newy = y+stepSize
            location = [x, y, newx, newy]
            newimg = img.crop(location)
            pict = [newimg, location]
            pictures.append(pict)
            print([x, y, newx, newy])
            # newimg.show()
    
    fig = plt.figure()
    columns = (x//200)+1
    rows = (y//200)+1
    print(len(pictures))
    print(columns, rows)
    for i in range(1, columns*rows +1):
        
        fig.add_subplot(rows, columns, i)
        plt.imshow(pictures[i-1][0])
        
    plt.show()

movingWindow(r"test.jpg")