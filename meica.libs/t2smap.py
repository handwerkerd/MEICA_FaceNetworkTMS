#!/usr/bin/env python

"""
# Multi-Echo ICA, Version 2.0
# See http://dx.doi.org/10.1016/j.neuroimage.2011.12.028
# Kundu, P., Inati, S.J., Evans, J.W., Luh, W.M. & Bandettini, P.A. Differentiating 
#	BOLD and non-BOLD signals in fMRI time series using multi-echo EPI. NeuroImage (2011).
#
# tedana.py version 2.0 	(c) 2012 Prantik Kundu, Noah Brenowitz, Souheil Inati
# tedana.py version 1.0		(c) 2012 Noah Brenowitz, Prantik Kundu, Souheil Inati
# tedana.py version 0.5		(c) 2011 Prantik Kundu, Souheil Inati
#
#Computes T2* map
"""

import os
from optparse import OptionParser
import numpy as np
import nibabel as nib
from sys import stdout


def niwrite(data,affine, name , header=None):
	stdout.write(" + Writing file: %s ...." % name) 
	
	thishead = header
	if thishead == None:
		thishead = head.copy()
		thishead.set_data_shape(list(data.shape))

	outni = nib.Nifti1Image(data,affine,header=thishead)
	outni.to_filename(name)
	print 'done.'

def cat2echos(data,Ne):
	"""
	cat2echos(data,Ne)

	Input:
	data shape is (nx,ny,Ne*nz,nt)
	"""
	nx,ny = data.shape[0:2]
	nz = data.shape[2]/Ne
	if len(data.shape) >3:
		nt = data.shape[3]
	else:
		nt = 1
	return np.reshape(data,(nx,ny,nz,Ne,nt),order='F')

def uncat2echos(data,Ne):
	"""
	uncat2echos(data,Ne)

	Input:
	data shape is (nx,ny,Ne,nz,nt)
	"""
    	nx,ny = data.shape[0:2]
	nz = data.shape[2]*Ne
	if len(data.shape) >4:
		nt = data.shape[4]
	else:
		nt = 1
	return np.reshape(data,(nx,ny,nz,nt),order='F')

def makemask(cdat):

	nx,ny,nz,Ne,nt = cdat.shape

	mask = np.ones((nx,ny,nz),dtype=np.bool)

	for i in range(Ne):
		tmpmask = (cdat[:,:,:,i,:] != 0).prod(axis=-1,dtype=np.bool)
		mask = mask & tmpmask

	return mask

def fmask(data,mask):
	"""
	fmask(data,mask)

	Input:
	data shape is (nx,ny,nz,...)
	mask shape is (nx,ny,nz)

	Output:
	out shape is (Nm,...)
	"""

	s = data.shape
	sm = mask.shape

	N = s[0]*s[1]*s[2]
	news = []
	news.append(N)

	if len(s) >3:
		news.extend(s[3:])

	tmp1 = np.reshape(data,news)
	fdata = tmp1.compress((mask > 0 ).ravel(),axis=0)

	return fdata.squeeze()

def unmask (data,mask):
	"""
	unmask (data,mask)

	Input:

	data has shape (Nm,nt)
	mask has shape (nx,ny,nz)

	"""
	M = (mask != 0).ravel()
	Nm = M.sum()

	nx,ny,nz = mask.shape

	if len(data.shape) > 1:
		nt = data.shape[1]
	else:
		nt = 1

	out = np.zeros((nx*ny*nz,nt),dtype=data.dtype)
	out[M,:] = np.reshape(data,(Nm,nt))

	return np.reshape(out,(nx,ny,nz,nt))

def t2smap(catd,mask,tes):
	"""
	t2smap(catd,mask,tes)

	Input:

	catd  has shape (nx,ny,nz,Ne,nt)
	mask  has shape (nx,ny,nz)
	tes   is a 1d numpy array
	"""
	nx,ny,nz,Ne,nt = catd.shape
	N = nx*ny*nz

	echodata = fmask(catd,mask)
	Nm = echodata.shape[0]

	#Do Log Linear fit
	B = np.reshape(np.abs(echodata), (Nm,Ne*nt)).transpose()
	B = np.log(B)
	x = np.array([np.ones(Ne),-tes])
	X = np.tile(x,(1,nt))
	X = np.sort(X)[:,::-1].transpose()

	beta,res,rank,sing = np.linalg.lstsq(X,B)
	t2s = 1/beta[1,:].transpose()
	s0  = np.exp(beta[0,:]).transpose()

	out = unmask(t2s,mask),unmask(s0,mask)

	return out

###################################################################################################
# 						Begin Main
###################################################################################################

if __name__=='__main__':

	parser=OptionParser()
	parser.add_option('-d',"--orig_data",dest='data',help="Spatially Concatenated Multi-Echo Dataset",default=None)
	parser.add_option('-e',"--TEs",dest='tes',help="Echo times (in ms) ex: 15,39,63",default=None)

	(options,args) = parser.parse_args()

	print "-- T2* Map Component for ME-ICA v2.0 --"

	if options.tes==None or options.data==None: 
		print "*+ Need at least data and TEs, use -h for help."		
		sys.exit()

	print "++ Loading Data"
	tes = np.fromstring(options.tes,sep=',',dtype=np.float32)
	ne = tes.shape[0]
	catim  = nib.load(options.data)	
	head   = catim.get_header()
	head.extensions = []
	head.set_sform(head.get_sform(),code=1)
	aff = catim.get_affine()
	catd = cat2echos(catim.get_data(),ne)
	nx,ny,nz,Ne,nt = catd.shape
	mu  = catd.mean(axis=-1)
	sig  = catd.std(axis=-1)
	
	print "++ Computing Mask"
	mask  = makemask(catd)

	print "++ Computing T2* map"
	t2s,s0   = t2smap(catd,mask,tes) 
	t2s[t2s>500] = 500

	niwrite(s0,aff,'s0v.nii')
	niwrite(t2s,aff,'t2sv.nii')
	
	

	


	




