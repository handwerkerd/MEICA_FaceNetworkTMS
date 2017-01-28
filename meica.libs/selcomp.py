import scipy.stats as stats
import numpy as np
import argparse
import ast
import sys
import os
"""
Tells you which components are in each bin of meica.py.
"""
def selcomps(seldict,orig,ne,var,debug=False,olevel=2,oversion=99,knobargs=''):

	#import ipdb

	#Dump dictionary into variable names
	for key in seldict.keys(): exec("%s=seldict['%s']" % (key,key))
	###########################################
	#List of components
	if orig or var:
		_varex_norm = varex_norm
		_varex = varex
		varex = _varex_norm
		varex_norm = _varex
	
	midk = []
	ign = []
	nc = np.arange(len(Kappas))
	ncl = np.arange(len(Kappas))

	#If user has specified 
	try:
		if options.manacc:
			acc = sorted([int(vv) for vv in options.manacc.split(',')])
			midk = []
			rej = sorted(np.setdiff1d(ncl,acc))
			return acc,rej,midk #Add string for ign
	except: 
		pass

	"""
	Set knobs
	"""
	LOW_PERC=25
	HIGH_PERC=90
	EXTEND_FACTOR=2
	try:
		if nt<100: EXTEND_FACTOR=3
	except: pass
	RESTRICT_FACTOR=2
	if knobargs!='': 
		for knobarg in ''.join(knobargs).split(','): exec(knobarg)

	"""
	Do some tallies for no. of significant voxels
	"""
	countsigZ = Z_clmaps.sum(0)
	countsigFS0 = F_S0_clmaps.sum(0)
	countsigFR2 = F_R2_clmaps.sum(0)
	countnoise = np.zeros(len(nc))

	"""
	Make table of dice values
	"""
	dice_table = np.zeros([nc.shape[0],2])
	for ii in ncl:
		dice_FR2 = tedana.dice(Br_clmaps_R2[:,ii],F_R2_clmaps[:,ii])
		dice_FS0 = tedana.dice(Br_clmaps_S0[:,ii],F_S0_clmaps[:,ii])
		dice_table[ii,:] = [dice_FR2,dice_FS0] #step 3a here and above
	dice_table[np.isnan(dice_table)]=0

	"""
	Make table of noise gain
	"""
	tt_table = np.zeros([len(nc),4])
	counts_FR2_Z = np.zeros([len(nc),2])
	for ii in nc:
		comp_noise_sel = tedana.andb([np.abs(Z_maps[:,ii])>1.95,Z_clmaps[:,ii]==0])==2
		countnoise[ii] = np.array(comp_noise_sel,dtype=np.int).sum()
		noise_FR2_Z = np.log10(np.unique(F_R2_maps[comp_noise_sel,ii]))
		signal_FR2_Z  = np.log10(np.unique(F_R2_maps[Z_clmaps[:,ii]==1,ii]))
		counts_FR2_Z[ii,:] = [len(signal_FR2_Z),len(noise_FR2_Z)]
		tt_table[ii,:2] = stats.ttest_ind(signal_FR2_Z,noise_FR2_Z,equal_var=False)
	tt_table[np.isnan(tt_table)]=0
	
	"""
	Assemble decision table
	"""
	d_table_rank = np.vstack([len(nc)-tedana.rankvec(Kappas), len(nc)-tedana.rankvec(dice_table[:,0]), \
		 len(nc)-tedana.rankvec(tt_table[:,0]), tedana.rankvec(countnoise), len(nc)-tedana.rankvec(countsigFR2) ]).T
	d_table_score = d_table_rank.sum(1)

	"""
	Step 1: Reject anything that's obviously an artifact
	a. Estimate a null variance
	"""
	rej = ncl[tedana.andb([Rhos>Kappas,countsigFS0>countsigFR2])>0]
	rej = np.union1d(rej,ncl[tedana.andb([dice_table[:,1]>dice_table[:,0],varex>np.median(varex)])==2])
	###########################################
	###########################################
	#removed following component selection criteria.  would remove high variance high kappa compnents sometimes if their rho was wasa little to high
	if orig:
		rej = np.union1d(rej,ncl[tedana.andb([tt_table[ncl,0]<0,varex[ncl]>np.median(varex)])==2])
	###########################################
	###########################################
	ncl = np.setdiff1d(ncl,rej)
	varex_ub_p = np.median(varex[Kappas>Kappas[tedana.getelbow(Kappas)]])

	
	"""
	Step 2: Make a  guess for what the good components are, in order to estimate good component properties
	a. Not outlier variance
	b. Kappa>kappa_elbow
	c. Rho<Rho_elbow
	d. High R2* dice compared to S0 dice
	e. Gain of F_R2 in clusters vs noise
	f. Estimate a low and high variance
	"""
	ncls = ncl.copy()
	###########################################
	###########################################
	# edited this section to be more lenient in accepting components as "good" 
	if orig:
		for nn in range(3): ncls = ncls[1:][(varex[ncls][1:]-varex[ncls][:-1])<varex_ub_p]
	else:
		for nn in range(3): ncls = np.union1d(ncls[1:][(varex[ncls][1:]-varex[ncls][:-1])<varex_ub_p],[ncls[0]]) #Step 2a, made this line automatically remove the highest kappa components 
	Kappas_lim = Kappas[Kappas<tedana.getfbounds(ne)[-1]]
	Rhos_lim = np.array(sorted(Rhos[ncls])[::-1])
	Rhos_sorted = np.array(sorted(Rhos)[::-1])
	Kappas_elbow = min(Kappas_lim[tedana.getelbow(Kappas_lim)],Kappas[tedana.getelbow(Kappas)])
	if orig:
		Rhos_elbow = np.mean([Rhos_lim[tedana.getelbow(Rhos_lim)]  , Rhos_sorted[tedana.getelbow(Rhos_sorted)], tedana.getfbounds(ne)[0]])#replaced these lines with the two below
	else:
		Rhos_elbow = Rhos_sorted[tedana.getelbow(Rhos_sorted)] - Rhos_sorted[tedana.getelbow(Rhos_sorted)]*0.05 # Use a more lenient elbow metric
	good_guess = ncls[tedana.andb([Kappas[ncls]>=Kappas_elbow, Rhos[ncls]<Rhos_elbow])==2]
	###########################################
	###########################################
	#End Ben Gutierrez edits
	
	if debug:
		import ipdb
		ipdb.set_trace()
	if len(good_guess)==0:
		return [],sorted(rej),[],sorted(np.setdiff1d(nc,rej))
	Kappa_rate = (max(Kappas[good_guess])-min(Kappas[good_guess]))/(max(varex[good_guess])-min(varex[good_guess]))
	Kappa_ratios = Kappa_rate*varex/Kappas
	varex_lb = tedana.scoreatpercentile(varex[good_guess],LOW_PERC )
	varex_ub = tedana.scoreatpercentile(varex[good_guess],HIGH_PERC)

	if debug:
		import ipdb
		ipdb.set_trace()

	"""
	Step 3: Get rid of midk components - those with higher than max decision score and high variance
	"""
	max_good_d_score = EXTEND_FACTOR*len(good_guess)*d_table_rank.shape[1]
	midkadd = ncl[tedana.andb([d_table_score[ncl] > max_good_d_score, varex[ncl] > EXTEND_FACTOR*varex_ub])==2]
	midk = np.union1d(midkadd, midk)
	ncl = np.setdiff1d(ncl,midk)

	"""
	Step 4: Find components to ignore
	"""
	good_guess = np.setdiff1d(good_guess,midk)
	loaded = np.union1d(good_guess, ncl[varex[ncl]>varex_lb])
	igncand = np.setdiff1d(ncl,loaded)
	igncand = np.setdiff1d(igncand, igncand[d_table_score[igncand]<max_good_d_score])
	igncand = np.setdiff1d(igncand,igncand[Kappas[igncand]>Kappas_elbow])
	ign = np.array(np.union1d(ign,igncand),dtype=np.int)
	ncl = np.setdiff1d(ncl,ign)

	if debug:
		import ipdb
		ipdb.set_trace()

	"""
	Step 5: Scrub the set
	"""

	if len(ncl)>len(good_guess):
		#Recompute the midk steps on the limited set to clean up the tail
		d_table_rank = np.vstack([len(ncl)-tedana.rankvec(Kappas[ncl]), len(ncl)-tedana.rankvec(dice_table[ncl,0]),len(ncl)-tedana.rankvec(tt_table[ncl,0]), tedana.rankvec(countnoise[ncl]), len(ncl)-tedana.rankvec(countsigFR2[ncl])]).T
		d_table_score = d_table_rank.sum(1)
		num_acc_guess = np.mean([np.sum(tedana.andb([Kappas[ncl]>Kappas_elbow,Rhos[ncl]<Rhos_elbow])==2), np.sum(Kappas[ncl]>Kappas_elbow)])
		candartA = np.intersect1d(ncl[d_table_score>num_acc_guess*d_table_rank.shape[1]/RESTRICT_FACTOR],ncl[Kappa_ratios[ncl]>EXTEND_FACTOR*2])
		midkadd = np.union1d(midkadd,np.intersect1d(candartA,candartA[varex[candartA]>varex_ub*EXTEND_FACTOR]))
		candartB = ncl[d_table_score>num_acc_guess*d_table_rank.shape[1]*HIGH_PERC/100.]
		midkadd = np.union1d(midkadd,np.intersect1d(candartB,candartB[varex[candartB]>varex_lb*EXTEND_FACTOR]))
		midk = np.union1d(midk,midkadd)
		#Find comps to ignore
		new_varex_lb = tedana.scoreatpercentile(varex[ncl[:num_acc_guess]],LOW_PERC)
		candart = np.setdiff1d(ncl[d_table_score>num_acc_guess*d_table_rank.shape[1]],midk)
		ignadd = np.intersect1d(candart,candart[varex[candart]>new_varex_lb])
		ignadd = np.union1d(ignadd,np.intersect1d(ncl[Kappas[ncl]<=Kappas_elbow],ncl[varex[ncl]>new_varex_lb]))
		ign = np.setdiff1d(np.union1d(ign,ignadd),midk)
		ncl = np.setdiff1d(ncl,np.union1d(midk,ign))

	if debug:
		import ipdb
		ipdb.set_trace()

	print('accepted components %s' % list(sorted(ncl)))
	print('rejected components %s' % list(sorted(rej)))
	print('midk components %s' % list(sorted([int(x) for x in midk])))#converting midk to int for asthetic reasons
	print('ign components %s' % list(sorted(ign)))
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Options')
	parser.add_argument('-path', dest = 'path', help = 'Path to meica.libs used')
	parser.add_argument('-echoes', dest = 'ne', help = 'number of echoes', default ='3',type = int)
	parser.add_argument('-orig', dest = 'orig',  help = 'Flag to use Prantik original component selection criteria', action = 'store_true')
	parser.add_argument('-var', dest = 'var',  help = 'swap variance and variance explained.  A reminder that the edited version of me-ica'
		+ ' does this automatically, so this is for seeing how the variance and variance explained effect component selection.', action = 'store_true')
	args = parser.parse_args()
	sys.path.append(args.path)
	import tedana
	names = ['F_S0_clmaps','varex','varex_norm','WTS','Kappas','Rhos','Z_clmaps','F_R2_maps','Z_maps','F_S0_maps','tsoc_B','Br_clmaps_S0','Br_clmaps_R2','F_R2_clmaps']
	variables = []
	for i in range(len(names)):
    		variables.append(np.loadtxt('%s.txt' % names[i]))
	F_S0_clmaps,varex,varex_norm,WTS,Kappas,Rhos,Z_clmaps,F_R2_maps,Z_maps,F_S0_maps,tsoc_B,Br_clmaps_S0,Br_clmaps_R2,F_R2_clmaps = variables[0],variables[1],variables[2],variables[3],variables[4],variables[5],variables[6],variables[7],variables[8],variables[9],variables[10],variables[11],variables[12],variables[13]	
	
	seldict = {}
	selvars = ['Kappas','Rhos','WTS','varex','varex_norm','Z_maps','F_R2_maps','F_S0_maps',\
			'Z_clmaps','F_R2_clmaps','F_S0_clmaps','tsoc_B','Br_clmaps_R2','Br_clmaps_S0']
	for vv in selvars:
		seldict[vv] = eval(vv)
	selcomps(seldict,args.orig,args.ne,args.var)