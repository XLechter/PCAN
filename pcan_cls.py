#This code is taken from Mika's PointnetVLAD

import tensorflow as tf
import numpy as np
import math
import sys
import os

import tf_util
from transform_nets import input_transform_net, feature_transform_net
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg
#Adopted from Antoine Meich
import tensorflow.contrib.slim as slim



def placeholder_inputs(batch_num_queries, num_pointclouds_per_query, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_num_queries, num_pointclouds_per_query, num_point, 3))
    return pointclouds_pl

#Adopted from the original pointnet code
def forward(point_cloud, is_training, bn_decay=None):
    """PointNetVLAD,    INPUT is batch_num_queries X num_pointclouds_per_query X num_points_per_pointcloud X 3, 
                        OUTPUT batch_num_queries X num_pointclouds_per_query X output_dim """
    batch_num_queries = point_cloud.get_shape()[0].value
    num_pointclouds_per_query = point_cloud.get_shape()[1].value
    num_points = point_cloud.get_shape()[2].value
    CLUSTER_SIZE=64
    OUTPUT_DIM=256
    point_cloud = tf.reshape(point_cloud, [batch_num_queries*num_pointclouds_per_query, num_points,3])

    point_cloud_xyz = point_cloud

    with tf.variable_scope('transform_net1') as sc:
        input_transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, input_transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    print ('input_image:', input_image)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        feature_transform = feature_transform_net(net, is_training, bn_decay, K=64)
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), feature_transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    print ('net:', net)

    net= tf.reshape(net,[-1,1024])
    net = tf.nn.l2_normalize(net,1)

    output, weights = vlad_forward(point_cloud_xyz, net, max_samples=num_points, is_training=is_training)


    print(output)

    #normalize to have norm 1
    output = tf.nn.l2_normalize(output,1)
    output =  tf.reshape(output,[batch_num_queries,num_pointclouds_per_query,OUTPUT_DIM])

    return output, weights


def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        #batch = query.get_shape()[0]
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1,int(num_pos),1]) #shape num_pos x output_dim
        best_pos=tf.reduce_min(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        #best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        return best_pos



##########Losses for PointNetVLAD###########

def SARE_loss(q_vec, pos_vecs, neg_vecs):
    num_pos = pos_vecs.get_shape()[1]
    query_copies_p = tf.tile(q_vec, [1, int(num_pos), 1])
    num_neg = neg_vecs.get_shape()[1]
    dif_p = -tf.reduce_sum(tf.squared_difference(pos_vecs, query_copies_p), 2)
    print('dif_p', dif_p)
    p_exp = tf.reduce_sum(tf.exp(dif_p), 1)
    #p_exp = tf.reduce_sum(p_exp, 1)
    print('p_exp', p_exp)
    query_copies_n = tf.tile(q_vec, [1, int(num_neg), 1])
    dif_n = -tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies_n), 2)
    print('dif_p', dif_n)
    n_exp = tf.reduce_sum(tf.exp(dif_n), 1)
    #n_exp = tf.reduce_sum(n_exp, 1)
    print('n_exp', n_exp)
    loss = tf.reduce_sum(-tf.log(tf.div(p_exp, (p_exp + n_exp))))
    return loss



#Returns average loss across the query tuples in a batch, loss in each is the average loss of the definite negatives against the best positive
def triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
     # ''', end_points, reg_weight=0.001):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss

#Lazy variant
def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss


def softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_sum(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    ones=tf.fill([int(batch), int(num_neg)],1.0)
    soft_loss=tf.reduce_mean(tf.reduce_max(tf.log(tf.exp(tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2)))+1.0),1))
    return soft_loss

def quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def lazy_quadruplet_loss_sm(q_vec, pos_vecs, neg_vecs, other_neg, m2):
    soft_loss= lazy_softmargin_loss(q_vec, pos_vecs, neg_vecs)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= soft_loss+second_loss

    return total_loss   

def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= triplet_loss(q_vec, pos_vecs, neg_vecs, m1)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    total_loss= trip_loss+second_loss

    return total_loss 

def lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss= lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)
    
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m2=tf.fill([int(batch), int(num_neg)],m2)

    second_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m2,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,other_neg_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))

    '''
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.001

    reg_loss = sum(reg_losses)*reg_constant
    '''
    #tv = tf.trainable_variables()
    '''
    for v in tv:
        print(type(v))
    '''
    #regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

    total_loss= trip_loss+second_loss

    return total_loss

def vlad_forward(xyz, reshaped_input, feature_size=1024, max_samples=4096, cluster_size=64,
                output_dim=256, gating=True, add_batch_norm=True,
                is_training=True, bn_decay=None):
    """Forward pass of a NetVLAD block.

    Args:
    reshaped_input: If your input is in that form:
    'batch_size' x 'max_samples' x 'feature_size'
    It should be reshaped in the following form:
    'batch_size*max_samples' x 'feature_size'
    by performing:
    reshaped_input = tf.reshape(input, [-1, features_size])

    Returns:
    vlad: the pooled vector of size: 'batch_size' x 'output_dim'
    """
    input = tf.reshape(reshaped_input, [-1,
                                    max_samples, feature_size])


    #msg grouping
    l1_xyz, l1_points = pointnet_sa_module_msg(xyz, input, 256, [0.1, 0.2, 0.4], [16, 32, 64],
                                               [[16, 16, 32], [32, 32, 64], [32, 64, 64]], is_training,
                                               bn_decay,
                                               scope='layer1', use_nchw=True)

    l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=None, radius=None, nsample=None,
                                              mlp=[256, 512], mlp2=None, group_all=True, is_training=is_training,
                                              bn_decay=bn_decay, scope='layer3')

    print('l2_points:', l2_points)

    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(xyz, l1_xyz, tf.concat([xyz,input],axis=-1), l1_points, [128,128], is_training, bn_decay, scope='fa_layer3')

    print('l0_points shape', l0_points)

    net = tf_util.conv1d(l0_points, 1, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 1, 1, padding='VALID', activation_fn=None, scope='fc2')

    m = tf.reshape(net, [-1, 1])
    print('m:', m)

    #constrain weights to [0, 1]
    m = tf.nn.sigmoid(m)
    weights = m
    m = tf.tile(m, [1, cluster_size])

    print('m:', m)

    cluster_weights = tf.get_variable("cluster_weights",
                                      [feature_size, cluster_size],
                                      initializer=tf.random_normal_initializer(
                                          stddev=1 / math.sqrt(feature_size)))

    activation = tf.matmul(reshaped_input, cluster_weights)

    # activation = tf.contrib.layers.batch_norm(activation,
    #         center=True, scale=True,
    #         is_training=self.is_training,
    #         scope='cluster_bn')

    # activation = slim.batch_norm(
    #       activation,
    #       center=True,
    #       scale=True,
    #       is_training=self.is_training,
    #       scope="cluster_bn")

    if add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn", fused=False)
    else:
        cluster_biases = tf.get_variable("cluster_biases",
                                         [cluster_size],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(feature_size)))
        activation = activation + cluster_biases

    activation = tf.nn.softmax(activation)

    activation_crn = tf.multiply(activation, m)

    activation = tf.reshape(activation_crn,
                            [-1, max_samples, cluster_size])

    a_sum = tf.reduce_sum(activation, -2, keepdims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
                                       [1, feature_size, cluster_size],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(feature_size)))

    a = tf.multiply(a_sum, cluster_weights2)

    activation = tf.transpose(activation, perm=[0, 2, 1])

    reshaped_input = tf.reshape(reshaped_input, [-1,
                                                 max_samples, feature_size])

    vlad = tf.matmul(activation, reshaped_input)
    vlad = tf.transpose(vlad, perm=[0, 2, 1])
    vlad = tf.subtract(vlad, a)

    vlad = tf.nn.l2_normalize(vlad, 1)

    vlad = tf.reshape(vlad, [-1, cluster_size * feature_size])
    vlad = tf.nn.l2_normalize(vlad, 1)

    hidden1_weights = tf.get_variable("hidden1_weights",
                                      [cluster_size * feature_size, output_dim],
                                      initializer=tf.random_normal_initializer(
                                          stddev=1 / math.sqrt(cluster_size)))

    ##Tried using dropout
    # vlad=tf.layers.dropout(vlad,rate=0.5,training=self.is_training)

    vlad = tf.matmul(vlad, hidden1_weights)

    ##Added a batch norm
    vlad = tf.contrib.layers.batch_norm(vlad,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope='bn')

    if gating:
        vlad = context_gating(vlad, add_batch_norm, is_training)

    return vlad, weights

def context_gating(input_layer, add_batch_norm=True, is_training=True):
    """Context Gating

    Args:
    input_layer: Input layer in the following shape:
    'batch_size' x 'number_of_activation'

    Returns:
    activation: gated layer in the following shape:
    'batch_size' x 'number_of_activation'
    """

    input_dim = input_layer.get_shape().as_list()[1]

    gating_weights = tf.get_variable("gating_weights",
                                     [input_dim, input_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(input_dim)))

    gates = tf.matmul(input_layer, gating_weights)

    if add_batch_norm:
        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            scope="gating_bn")
    else:
        gating_biases = tf.get_variable("gating_biases",
                                        [input_dim],
                                        initializer=tf.random_normal(stddev=1 / math.sqrt(input_dim)))
        gates = gates + gating_biases

    gates = tf.sigmoid(gates)

    activation = tf.multiply(input_layer, gates)

    return activation
