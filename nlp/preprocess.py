# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np


class embedding(object):

    def __init__(self,word_vec_size=200):
        self.word_vec_size=word_vec_size

    def load_word_vectors(self,vec_path):
        '''load binary file of word vectors with the C format'''
        vocab=dict()
        with open(vec_path,'rb') as f:
            header=unicode(f.readline(),encoding='utf8')
            vocab_size,vector_size=map(int,header.split())
            binary_len=np.dtype(np.float32).itemsize*vector_size
            for line in np.arange(vocab_size):
                word=[]
                while True:
                    ch=f.read(1)
                    if ch==b' ':
                        break
                    if ch==b'':
                        raise EOFError('unexpected end of input')
                    if ch!=b'\n':
                        word.append(ch)
                word=unicode(''.join(word),encoding='utf8')
                weights=np.fromstring(f.read(binary_len),dtype=np.float32)
                vocab[word]=weights
        return vocab_size,vector_size,vocab

    def load_sentence_vector(self,sent,action,word_vec_size,sep=','):
        sent=np.fromstring(sent,dtype=np.float32,sep=sep)
        if action=='train':
            return np.int8(sent[0]),sent[1:].reshape((-1,word_vec_size))
        elif action=='classify':
            return 0,sent.reshape((-1,word_vec_size))
        else:
            raise ValueError('use train or classify for action')

    def load_sentence_matrix(self,sent_path,action='train',word_vec_size=None):
        if not word_vec_size:
            word_vec_size=self.word_vec_size
        with open(sent_path,'r') as f:
            print('loading %s' % sent_path)
            lines=f.readlines()
        return zip(*(self.load_sentence_vector(l,action,word_vec_size) for l in lines))


if __name__=='__main__':

    pass
