# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tensorflow as tf #r1.0
import numpy as np
import time
from preprocess import embedding

def read_argvs():
    import argparse
    parser=argparse.ArgumentParser(description=
            '1) Convolutional Neural Networks for Sentence Classification (arxiv.org/abs/1408.5882)\n'+
            '2) A Sensitivity Analysis of (and Practitioners Guide to) Convolutional Neural Networks for Sentence Classification (arxiv.org/abs/1510.03820)',
            formatter_class=argparse.RawTextHelpFormatter,add_help=True)
    parser.add_argument('--action',type=str,required=False,metavar='train',default='train',
            help="'train' or 'classify'")
    parser.add_argument('--improved',required=False,action='store_true',default=False,
            help="one more hidden layer added in the full connected part")
    parser.add_argument('--num_fc_nodes',type=int,required=False,metavar=120,default=120,
            help='number of nodes in the hidden layer added in the full connected part of the improved model')
    parser.add_argument('--vector_path',type=str,required=False,metavar='vector.bin',default='vector.bin',
            help='word embeddings from word2vec')
    parser.add_argument('--train_path',type=str,required=False,metavar='train.dat',default='train.dat',
            help='train file for training')
    parser.add_argument('--test_path',type=str,required=False,metavar='test.dat',default='test.dat',
            help='test file for training')
    parser.add_argument('--eval_path',type=str,required=False,metavar='eval.dat',default='eval.dat',
            help='file for evaluation')
    parser.add_argument('--restart',required=False,action='store_true',default=False,
            help='restart training')
    parser.add_argument('--model_path',type=str,required=False,metavar='./model.ckpt',default='./model.ckpt',
            help='model parameters')
    parser.add_argument('--out_path',type=str,required=False,metavar='eval.out',default='eval.out',
            help='result of classification')
    parser.add_argument('--batch_size',type=int,required=False,metavar=128,default=128,
            help='batch size')
    parser.add_argument('--filter_sizes',type=str,required=False,metavar='2,3,4',default='2,3,4',
            help='sizes of convolutional filters')
    parser.add_argument('--filter_number',type=int,required=False,metavar=40,default=40,
            help='number of filters per filter size')
    parser.add_argument('--dropout_keep_prob',type=float,required=False,metavar=0.7,default=0.7,
            help='keep probability of dropout')
    parser.add_argument('--sentence_max_length',type=int,required=False,metavar=150,default=150,
            help='max length of a sentence')
    parser.add_argument('--word_vector_length',type=int,required=False,metavar=200,default=200,
            help='length of word embedding vector')
    parser.add_argument('--class_number',type=int,required=False,metavar=3,default=3,
            help='number of classes')
    parser.add_argument('--train_step_number',type=int,required=False,metavar=100,default=100,
            help='number of training steps')
    parser.add_argument('--learning_rate',type=float,required=False,metavar=1e-4,default=1e-4,
            help='learning rate for optimizer')
    parser.add_argument('--l2_weight',type=float,required=False,metavar=1e-4,default=1e-4,
            help='weight of L2 norm constrain')
    return parser.parse_args()


class cnn_Kim2014(object):

    def __init__(self,sent_max_length=150,word_vec_size=200,filter_sizes=[2,3,4],num_filters=40,
            dropout_keep_prob=0.7,batch_size=128,num_classes=3,
            is_improved_model=True,num_fc_hidden_nodes=120,weight_path='./model.ckpt'):
        self.sent_max_length=sent_max_length
        self.word_vec_size=word_vec_size
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.dropout_prob=dropout_keep_prob
        self.batch_size=batch_size
        self.num_classes=num_classes
        self.is_improved_model=is_improved_model
        self.num_fc_hidden_nodes=num_fc_hidden_nodes
        self.weight_path=weight_path
        self.seed=1234
    
    def _init_weights(self,name,w_shape,b_shape):
        if 'conv' in name:
            initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=self.seed)
        elif 'fc' in name:
            initializer=tf.contrib.layers.xavier_initializer(seed=self.seed)
        w=tf.get_variable(name,shape=w_shape,dtype=tf.float32,initializer=initializer,trainable=True)
        b=tf.Variable(tf.constant(0.1,shape=b_shape),trainable=True,dtype=tf.float32)
        return w,b

    def _conv_block(self,g,w,b,w_height,w_width,g_height):
        g=tf.nn.conv2d(g,w,[1,1,w_width,1],padding='VALID')
        g=tf.nn.relu(tf.nn.bias_add(g,b))
        g=tf.nn.max_pool(g,ksize=[1,g_height-w_height+1,1,1],strides=[1,1,1,1],padding='VALID')
        return g

    def build_cnn(self,l2_reg_lambda):
        sent_matrix=tf.placeholder(dtype=tf.float32,shape=[None,self.sent_max_length,self.word_vec_size])
        sent_matrix_expand=tf.expand_dims(sent_matrix,-1)
        labels=tf.placeholder(dtype=tf.float32,shape=[None,self.num_classes])
        graph=[]
        l2_loss=tf.constant(0.0)
        for filter_size in self.filter_sizes:  
            w,b=self._init_weights('w_conv_'+str(filter_size),
                    [filter_size,self.word_vec_size,1,self.num_filters],[self.num_filters])
            l2_loss+=tf.nn.l2_loss(w)
            l2_loss+=tf.nn.l2_loss(b)
            block=self._conv_block(sent_matrix_expand,w,b,
                    filter_size,self.word_vec_size,self.sent_max_length)
            graph.append(block)
        graph=tf.concat(graph,3)
        graph=tf.squeeze(graph) # (batch_size,filter_size*num_filters)
        graph=tf.nn.dropout(graph,self.dropout_prob)
        w,b=self._init_weights('w_fc',[self.num_filters*len(self.filter_sizes),self.num_classes],
                [self.num_classes])
        l2_loss+=tf.nn.l2_loss(w)
        l2_loss+=tf.nn.l2_loss(b)
        scores=tf.nn.xw_plus_b(graph,w,b)
        predictions=tf.argmax(scores,1)
        accuracy=tf.equal(predictions,tf.argmax(labels,1))
        accuracy=tf.reduce_mean(tf.cast(accuracy,tf.float32))
        loss=tf.nn.softmax_cross_entropy_with_logits(logits=scores,labels=labels)
        loss=tf.reduce_mean(loss)+l2_reg_lambda*l2_loss
        return sent_matrix,labels,predictions,accuracy,loss
        
    def build_cnn_revised(self,l2_reg_lambda):
        sent_matrix=tf.placeholder(dtype=tf.float32,shape=[None,self.sent_max_length,self.word_vec_size])
        sent_matrix_expand=tf.expand_dims(sent_matrix,-1)
        labels=tf.placeholder(dtype=tf.float32,shape=[None,self.num_classes])
        graph=[]
        l2_loss=tf.constant(0.0)
        for filter_size in self.filter_sizes:
            w,b=self._init_weights('w_conv_'+str(filter_size),
                    [filter_size,self.word_vec_size,1,self.num_filters],[self.num_filters])
            l2_loss+=tf.nn.l2_loss(w)
            l2_loss+=tf.nn.l2_loss(b)
            block=self._conv_block(sent_matrix_expand,w,b,
                    filter_size,self.word_vec_size,self.sent_max_length)
            graph.append(block)
        graph=tf.concat(graph,3)
        graph=tf.squeeze(graph) # (batch_size,filter_size*num_filters)
        w,b=self._init_weights('w_fc_1',[self.num_filters*len(self.filter_sizes),self.num_fc_hidden_nodes],[self.num_fc_hidden_nodes])
        l2_loss+=tf.nn.l2_loss(w)
        l2_loss+=tf.nn.l2_loss(b)
        graph=tf.nn.xw_plus_b(graph,w,b)
        graph=tf.nn.relu(graph)
        graph=tf.nn.dropout(graph,self.dropout_prob)
        w,b=self._init_weights('w_fc_2',[self.num_fc_hidden_nodes,self.num_classes],[self.num_classes])
        l2_loss+=tf.nn.l2_loss(w)
        l2_loss+=tf.nn.l2_loss(b)
        scores=tf.nn.xw_plus_b(graph,w,b)
        predictions=tf.argmax(scores,1)
        accuracy=tf.equal(predictions,tf.argmax(labels,1))
        accuracy=tf.reduce_mean(tf.cast(accuracy,tf.float32))
        loss=tf.nn.softmax_cross_entropy_with_logits(logits=scores,labels=labels)
        loss=tf.reduce_mean(loss)+l2_reg_lambda*l2_loss
        return sent_matrix,labels,predictions,accuracy,loss

    def mini_batch(self,x,y,sent_max_length,word_vec_size,batch_size,num_classes,is_shuffle=True):
        x=np.array(x)
        y=np.array(y)
        assert (len(x)==len(y)), 'the lenght of x is not equal to that of y'
        num_batch=len(y)//batch_size # discard the tail
        if is_shuffle:
            permu=np.arange(len(y))
            np.random.shuffle(permu)
            x=x[permu]
            y=y[permu]
        x_batch,y_batch=[],[]
        x_=np.empty(shape=(len(x),sent_max_length,word_vec_size))
        y_=np.zeros(shape=(len(y),num_classes))
        for i in np.arange(num_batch):
            for j in np.arange(batch_size):
                if len(x[i*batch_size+j])<=sent_max_length:
                    x_[i*batch_size+j]=np.pad(x[i*batch_size+j],
                            ((0,sent_max_length-len(x[i*batch_size+j])),(0,0)),'constant',constant_values=0)
                else:
                    x_[i*batch_size+j]=x[i*batch_size+j][:sent_max_length]
                y_[i*batch_size+j,y[i*batch_size+j]-1]=1
            x_batch.append(x_[(i*batch_size):((i+1)*batch_size)])
            y_batch.append(y_[(i*batch_size):((i+1)*batch_size)])
        return x_batch,y_batch

    def train_cnn(self,x_train,y_train,x_test,y_test,
            nstep=50,learning_rate=1e-4,l2_reg_lambda=1e-4,restart=False):
        x_batch,y_batch=self.mini_batch(x_train,y_train,self.sent_max_length,self.word_vec_size,
                self.batch_size,self.num_classes,True)
        x_test,y_test=self.mini_batch(x_test,y_test,self.sent_max_length,self.word_vec_size,
                len(y_test),self.num_classes,False)
        with tf.Graph().as_default(),tf.Session(
                config=tf.ConfigProto(log_device_placement=False)) as sess:
            with tf.device('/gpu:0'):
                if self.is_improved_model:
                    print('using the revised model')
                    sent_batch,label_batch,predictions,accuracy,loss=self.build_cnn_revised(l2_reg_lambda)
                else:
                    sent_batch,label_batch,predictions,accuracy,loss=self.build_cnn(l2_reg_lambda)
                optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
            with tf.device('/cpu:0'):
                saver=tf.train.Saver() # works only on cpu
            if restart:
                saver.restore(sess,self.weight_path)
            else:
                sess.run(tf.global_variables_initializer())

            for step in np.arange(nstep):
                start_time=time.time()
                loss_train,accuracy_train=[],[]
                for i in np.arange(len(y_batch)):
                    _,accuracy_,loss_=sess.run([optimizer,accuracy,loss],
                            feed_dict={sent_batch:x_batch[i],label_batch:y_batch[i]})
                    loss_train.append(loss_)
                    accuracy_train.append(accuracy_)
                accuracy_train=np.mean(accuracy_train)
                loss_train=np.mean(loss_train)
                accuracy_test=sess.run(accuracy,feed_dict={sent_batch:x_test[0],label_batch:y_test[0]})
                duration=time.time()-start_time
                print('epoch %d: loss %.3f, train_acc %.3f, test_acc %.3f (%.2f sec)' % 
                        (step+1,loss_train,accuracy_train,accuracy_test,duration))
            save_path=saver.save(sess,self.weight_path)
            print('Model is saved in %s' % save_path)
        return 

    def classify_sentences(self,x,fake_y,out_path='eval.dat'):
        start_time=time.time()
        x,fake_y=self.mini_batch(x,fake_y,self.sent_max_length,self.word_vec_size,
                len(fake_y),self.num_classes,False)
        with tf.Graph().as_default(),tf.Session(
                config=tf.ConfigProto(log_device_placement=False)) as sess:
            with tf.device('/gpu:0'):
                if self.is_improved_model:
                    print('using the revised model')
                    sents,fake_labels,predictions,_,_=self.build_cnn_revised(0.0)
                else:
                    sents,fake_labels,predictions,_,_=self.build_cnn(0.0)
            with tf.device('/cpu:0'):
                tf.train.Saver().restore(sess,self.weight_path)
            start_time=time.time()
            pred_values=sess.run(predictions,feed_dict={sents:x[0],fake_labels:fake_y[0]})
        pred_values=np.add(pred_values,1)
        np.savetxt(out_path,pred_values,fmt='%i',delimiter=',')
        duration=time.time()-start_time
        print('Results are dumped into %s (%.1f sec)' % (out_path,duration))
        return


if __name__=='__main__':

    FLAGS=read_argvs()
    FLAGS.filter_sizes=map(int,FLAGS.filter_sizes.strip().split(','))
    inputs=embedding(FLAGS.word_vector_length)
    model=cnn_Kim2014(sent_max_length=FLAGS.sentence_max_length,word_vec_size=FLAGS.word_vector_length,
            filter_sizes=FLAGS.filter_sizes,num_filters=FLAGS.filter_number,
            dropout_keep_prob=FLAGS.dropout_keep_prob,batch_size=FLAGS.batch_size,num_classes=FLAGS.class_number,
            is_improved_model=FLAGS.improved,num_fc_hidden_nodes=FLAGS.num_fc_nodes,weight_path=FLAGS.model_path)

    if FLAGS.action.lower()=='train':
        train_labels,train_sents=inputs.load_sentence_matrix(FLAGS.train_path)
        test_labels,test_sents=inputs.load_sentence_matrix(FLAGS.test_path)
        model.train_cnn(x_train=train_sents,y_train=train_labels,x_test=test_sents,y_test=test_labels,
                nstep=FLAGS.train_step_number,learning_rate=FLAGS.learning_rate,restart=FLAGS.restart,
                l2_reg_lambda=FLAGS.l2_weight)
    elif FLAGS.action.lower()=='classify':
        fake_labels,sents=inputs.load_sentence_matrix(FLAGS.eval_path)
        model.classify_sentences(x=sents,fake_y=fake_labels,out_path=FLAGS.out_path)
    else:
        print("use 'train' or 'classify' for the argument --action")

