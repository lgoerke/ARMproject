import flickrapi
import urllib.request
import os

# Application key for accessing the Flickr-API
key = "f068714e6ddffb793ef202e551d14f8f"
secret = "d8f34d5bfce34217"

# Create API-instance
flickr = flickrapi.FlickrAPI(key, secret)

# Define terms for image search on Flickr
# Terms should be sensible Categories and available Classes from ImageNet
tags = ["table", "car", "airplane", "church", "fruit", "house", "dog", "cat", "teapot", "table lamp", "castle", "pillow", "volcano", "coffee mug", "envelope"]
# Download images per term
for tag in tags:
    
    # Create folder to store images in
    if not os.path.exists('data/'+tag):
        os.mkdir('data/'+tag)
    
    # Iterate through search results ordered by relevance
    for i,photo in enumerate(flickr.walk(tag_mode='all',
            text=tag,sort="relevance")):
        
        if i > 500: break; # Stop after 50 images
        # Download the image    
        img_url = "http://farm%s.static.flickr.com/%s/%s_%s_m.jpg" % (photo.get('farm'), photo.get('server'), photo.get('id'), photo.get('secret'))
        # Save images to folder
        with open('data/'+tag + "/0000000" + str(i) + ".jpg", "wb") as f:
            with urllib.request.urlopen(img_url) as url:
                f.write(url.read())
