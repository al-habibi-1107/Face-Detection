#header files required
from skimage.feature import Cascade
from skimage import data
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#Getting the image 
path=''
img= plt.imread(path)
plt.axis('off')
plt.imshow(img)

#reading the dataset for facial detection
train_set= data.lbp_frontal_face_cascade_filename()
detector= Cascade(train_set)
detected = detector.detect_multi_scale(img=img,
                                      scale_factor=1.2,
                                      step_ratio=1,
                                      min_size=(10,10),
                                      max_size=(200,200) 
                                      )
#function to draw a rectangle around the detected faces
def show_detected_face(result,detected,title='Detected Faces'):
    plt.imshow(result)
    desc=plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')
    for rec in detected:
        desc.add_patch(
            Rectangle(
                (rec['c'],rec['r']),
                rec['width'],
                rec['height'],
                Fill=False,color='g',linewidth=2)
                )
    plt.show()
    
#calling the function
show_detected_face(img,detected)






