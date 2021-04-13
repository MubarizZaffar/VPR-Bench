# Hog-feature
HOG feature extractor with simple python implementation

# Running

``` python
# hog.py
img = cv2.imread('data/picture1.png', cv2.IMREAD_GRAYSCALE)
hog = Hog_descriptor(img, cell_size=8, bin_size=8)
vector, image = hog.extract()
plt.imshow(image, cmap=plt.cm.gray)
plt.show()
```

# Result

<img src="/data/picture1.png" width="700">
<img src="/figure_1.png" width="700">

<img src="/data/picture2.png" width="700">
<img src="/figure_2.png" width="700">

<img src="/person.png" width="700">
