import cv2
import tensorflow as tf
from tensorflow import nn
import numpy as np

#Localization parameters
DET_SIZE = (300,300)    #Run all localization at a standard size
IN_SIZE = (32,32)   #Input dimensions of image for the network
BLUR_DIM = (50,50)      #Dimension for blurring the face location mask
CONF_THRESH = 0.99      #Confidence threshold to mark a window as a face

X_STEP = 10     #Horizontal slide for the sliding window
Y_STEP = 10     #Vertical stride for the sliding window
WIN_MIN = 40    #Minimum sliding window size
WIN_MAX = 100   #Maximum sliding window size
WIN_STRIDE = 10   #Stride to increase the sliding window


#Make weight and bias variables -- From the TensorFlow tutorial
def weight(shape):
    intial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial)


def bias(shape):
    intial = tf.constant(0.1, shape=shape)
    return tf.Variable(intial)


#Finds the product of a dimension tuple to find the total legth
def dim_prod(dim_arr):
    return np.prod([d for d in dim_arr if d != None])


#Start a TensorFlow session
def start_sess():
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    return sess


#Build the net in the session
def build_net(sess):
    in_len = 32
    in_dep = 1

    x_hold = tf.placeholder(tf.float32,shape=[None,in_dep*in_len*in_len])
    y_hold = tf.placeholder(tf.float32,shape=[None,2])
    keep_prob = tf.placeholder(tf.float32)

    xt = tf.reshape(x_hold,[-1,in_len,in_len,in_dep])

    #Layer 1 - 5x5 convolution
    w1 = weight([5,5,in_dep,4])
    b1 = bias([4])
    c1 = nn.relu(nn.conv2d(xt,w1,strides=[1,2,2,1],padding='VALID')+b1)
    o1 = c1

    #Layer 2 - 3x3 convolution
    w2 = weight([3,3,4,16])
    b2 = bias([16])
    c2 = nn.relu(nn.conv2d(o1,w2,strides=[1,2,2,1],padding='VALID')+b2)
    o2 = c2

    #Layer 3 - 3x3 convolution
    w3 = weight([3,3,16,32])
    b3 = bias([32])
    c3 = nn.relu(nn.conv2d(o2,w3,strides=[1,1,1,1],padding='VALID')+b3)
    o3 = c3

    dim = 32 * 4*4
        
    #Fully connected layer - 600 units
    of = tf.reshape(o3,[-1,dim])
    w4 = weight([dim,600])
    b4 = bias([600])
    o4 = nn.relu(tf.matmul(of,w4)+b4)

    o4 = nn.dropout(o4, keep_prob)

    #Output softmax layer - 2 units
    w5 = weight([600,2])
    b5 = bias([2])
    y = nn.softmax(tf.matmul(o4,w5)+b5)

    sess.run(tf.global_variables_initializer())

    return y,x_hold,y_hold,keep_prob


#Basic sliding window detector to find faces
#Returns an image showing only the faces along with the sliding window mask (before blurring)
def localize(img,model_path):
    sess = start_sess()
    y,x_hold,y_hold,keep_prob = build_net(sess)
    saver = tf.train.Saver()
    saver.restore(sess,model_path)

    #Run all detection at a fixed size
    img = cv2.resize(img,DET_SIZE)
    mask = np.zeros(img.shape)
    #Run sliding windows of different sizes
    for bx in range(WIN_MIN,WIN_MAX,WIN_STRIDE):
        by = bx
        for i in range(0, img.shape[1]-bx, X_STEP):
            for j in range(0, img.shape[0]-by, Y_STEP):
                sub_img = cv2.resize(img[i:i+bx,j:j+by],IN_SIZE)
                X = sub_img.reshape((1,dim_prod(IN_SIZE)))
                out = y.eval(session=sess,feed_dict={x_hold:X,keep_prob:1})[0]
                if out[0] >= CONF_THRESH:
                    mask[i:i+bx,j:j+by] = mask[i:i+bx,j:j+by]+1

    sess.close()
    mask = np.uint8(255*mask/np.max(mask))
    faces = img*(cv2.threshold(cv2.blur(mask,BLUR_DIM),0,255,cv2.THRESH_OTSU)[1]/255)
    #return (faces,mask)

    #Return image
    ymin = 300
    ymax = 0
    xmin = 300
    xmax = 0
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if mask[i,j] != 0:
                if ymin > i:
                    ymin = i
                if ymax < i:
                    ymax = i
                if xmin > j:
                    xmin = j
                if xmax < j:
                    xmax = j

    #print(str(ymin) + str(ymax) + str(xmin) + str(xmax))
    face_img = img[ymin:ymax, xmin:xmax]
    #cv2.imshow("Face", face_img)
    return face_img
