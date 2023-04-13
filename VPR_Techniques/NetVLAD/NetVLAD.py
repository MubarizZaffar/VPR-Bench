import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from . import netvlad_tf.net_from_mat as nfm
from . import netvlad_tf.nets as nets
import time

def compute_map_features(ref_map_images):
    
    ref_desc=[]
    tf.reset_default_graph()
    
    image_batch = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, 3])
    
    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.train.Saver()
    
    sess = tf.Session()
    saver.restore(sess, nets.defaultCheckpoint())
    
    for img in ref_map_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640,480), interpolation=cv2.INTER_CUBIC)
        batch = np.expand_dims(img, axis=0)
        t1=time.time()
        desc = sess.run(net_out, feed_dict={image_batch: batch})#[0][0:1024]
        print(('Encode Time: ', time.time()-t1))
        ref_desc.append(desc)
        
        print((desc.shape))
        
    return ref_desc

def compute_query_desc(image_query):
    image_query=cv2.resize(image_query, (640,480), interpolation=cv2.INTER_CUBIC)
    tf.reset_default_graph()
    
    image_batch = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, 3])
    
    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.train.Saver()
    
    sess = tf.Session()
    saver.restore(sess, nets.defaultCheckpoint())
    
    batch = np.expand_dims(image_query, axis=0)
    query_desc = sess.run(net_out, feed_dict={image_batch: batch})#[0][0:1024] 
    print((query_desc.shape)) 
    
    return query_desc
       
def perform_VPR(query_desc,ref_map_features):
    all_scores=[]
    for i in range(len(ref_map_features)):
        t1=time.time()    
        query_desc=query_desc.astype('float64')
        ref_desc=ref_map_features[i].astype('float64')
        match_score=np.dot(query_desc,ref_desc.T)
        t2=time.time()
        print(('NetVLAD tm:',t2-t1))
        all_scores.append(match_score)
    
    return np.amax(all_scores), np.argmax(all_scores),  np.asarray(all_scores).reshape(len(ref_map_features))
        
    