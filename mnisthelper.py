from struct import unpack
import gzip
from numpy import zeros, uint8
from pylab import imshow, show, cm

def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()

def get_labeled_data(imagefile, labelfile,samplesize):
    """Read input-vector (image) and target class (label, 0-9) and return
    it as list of tuples.
    """
    print "Reading Labled Data (n=%s) from Image: %s Label: %s" % (samplesize,imagefile,labelfile)
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')
    # Read the binary data
    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]
    
    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    number_of_labels = labels.read(4)
    number_of_labels = unpack('>I', number_of_labels)[0]
    
    if number_of_images != number_of_labels:
        raise Exception('number of labels did not match the number of images')
    if(samplesize>number_of_images):
        print "given sample size exceeds the size of the dataset %s observations." % (number_of_images)
        print "setting sample size to %s" % (number_of_images)
        N=number_of_images
    else:
        N=samplesize
        # Get the data
        x = zeros((N, rows, cols), dtype=uint8)	# Initialize numpy array
        y = zeros((N, 1), dtype=uint8)		# Initialize numpy array
        for i in range(N):
            for row in range(rows):
                for col in range(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    x[i][row][col] = tmp_pixel
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]
        print "Processed %s datapoints."%(N)
        return (x, y)

