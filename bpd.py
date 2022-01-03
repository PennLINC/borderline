#!/cbica/home/bertolem/anaconda3/bin/python
from numpy.lib import utils
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import scipy
from itertools import combinations

import pennlinckit.brain
import pennlinckit.utils
import pennlinckit.data
import pennlinckit.network

from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pylab as plt
import sys
import os
from pennlinckit import plotting
import glob
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from sklearn.kernel_ridge import KernelRidge
from multiprocessing import Pool
import leidenalg as la
import igraph as ig
global homedir
import time
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

homedir = '/cbica/home/bertolem/bpd/'


def score_bdp(data):
	if data.source == 'hcpd-dcan':
		reverse = [40,5,50,31,8,33]
		regular = [14,39,54,9,59,24,29,45,55,22,36,21,26,41,51,11,28,30]
		neo_name = 'nffi'

	if data.source == 'hcpya':
		reverse = ['40','05','50','31','33','08']
		regular = ['14','39','54','09','59','24','29','45','55','22','36','21','26','41','51','11','28','30']
		neo_name = 'NEORAW'

	def apply_dict(dict,x):
		return np.array(list(map(dict.get,x)))
	val_dict = {'SA':5,'A':4,'N':3,'D':2,'SD':1,'nan':np.nan}

	assert len(np.intersect1d(reverse,regular)) == 0
	assert len(np.unique(reverse)) + len(np.unique(regular)) ==24

	bpd_score = []
	for i in regular:
		x = data.subject_measures['%s_%s'%(neo_name,i)].values
		if data.source == 'hcpya':
			x = apply_dict(val_dict,x.astype(str))
		bpd_score.append(x)
	for i in reverse:
		x = data.subject_measures['%s_%s'%(neo_name,i)].values
		if data.source == 'hcpya':
			x = apply_dict(val_dict,x.astype(str))
		x = x * -1
		bpd_score.append(x)
	bpd_score = np.nanmean(bpd_score,axis=0)
	data.subject_measures['bpd_score'] = bpd_score

def make_data(source,cores=10):
	"""
	Make the datasets and run network metrics
	"""

	data = pennlinckit.data.dataset(source,task='**', parcels='Schaefer417',fd_scrub=.2)
	data.load_matrices()
	data.filter(way='>',value=100,column='n_frames')

	if source == 'hcpd-dcan':
		gender = np.zeros((data.subject_measures.shape[0]))
		gender[data.subject_measures.sex=='F'] = 1
		data.subject_measures['gender_dummy'] = gender	

	if source == 'hcpya':
		gender = np.zeros((data.subject_measures.shape[0]))
		gender[data.subject_measures.Gender=='F'] = 1
		data.subject_measures['gender_dummy'] = gender

	if source == 'pnc':
		data.subject_measures['gender_dummy'] = data.subject_measures.sex.values - 1

	if source == 'hcpya' or source == 'hcpd-dcan' or source == 'nki':
		score_bdp(data)
	data.network = pennlinckit.network.make_networks(data,yeo_partition=7,cores=cores-1)
	pennlinckit.utils.save_dataset(data,'/{0}/data/{1}.data'.format(homedir,source))

def submit_make_data(source):
	"""
	The above function makes the datasets (including the networks) we are going to use to generate uncomment out the code below and
	"""
	script_path = '/cbica/home/bertolem/bpd/bpd.py make_data {0}'.format(source) #it me
	pennlinckit.utils.submit_job(script_path,'d_{0}'.format(source),RAM=40,threads=10)

def load_data(source,filters=['bpd_score','meanFD']):
	data = pennlinckit.utils.load_dataset('/{0}/data/{1}.data'.format(homedir,source))
	gender = np.zeros((data.subject_measures.shape[0]))
	if source == 'hcpya':
		gender[data.subject_measures.Gender=='F'] = 1
	if source == 'hcpd-dcan':
		gender[data.subject_measures.sex=='F'] = 1
	data.subject_measures['gender_dummy'] = gender
	for f in filters:
		data.filter(way='has_subject_measure',value=f)
	return data

def dumb_predict():
	adult = pennlinckit.utils.load_dataset('/{0}/data/{1}.matrices'.format(homedir,'hcpya'))
	dev = pennlinckit.utils.load_dataset('/{0}/data/{1}.data'.format(homedir,'hcpd-dcan'))
	adult_bpd_brain = adult.network.pc[adult.subject_measures.bpd_score>1].mean(axis=0).mean(axis=0)

	prediction = []

	for d in range(dev.matrix.shape[0]):
		r = pennlinckit.utils.nan_pearsonr(dev.network.pc[d].mean(axis=0),adult_bpd_brain)[0]
		prediction.append(r)
		if d >5:
			print (pearsonr(prediction[:d],dev.subject_measures.bpd_score.values[:d]))

def apply_ya_2_dev(source='full',dev_data='hcpd-dcan',remove_linear_vars= ['gender_dummy','meanFD','interview_age'],hcpya_remove_linear_vars= ['gender_dummy','meanFD']):
	regress_name = '_'.join(remove_linear_vars)
	
	dev = pennlinckit.utils.load_dataset('/{0}/data/{1}.data'.format(homedir,dev_data))
	ya = pennlinckit.utils.load_dataset('/{0}/data/{1}.data'.format(homedir,'hcpya'))

	ya.filter(way='has_subject_measure',value='bpd_score')
	dev.filter(way='has_subject_measure',value='meanFD')

	if source == 'same': # get the lowest motion subjects to match the size of hcpd
		bpd_score_mask = np.isnan(dev.subject_measures.bpd_score.values)
		fold_length = len(bpd_score_mask[bpd_score_mask==False])
		match_size_mean_FD = ya.subject_measures.meanFD.values[np.argsort(ya.subject_measures.meanFD)][fold_length+1]    
		ya.filter('<',match_size_mean_FD,"meanFD") 
	
	else: # just get low motion subjects
		ya.filter('<',.2,"meanFD") 

	"""
	fit the model to predict bpd in hcpya
	"""

	ya.targets = ya.subject_measures['bpd_score'].values
	models = []
	acc = []
	for node in range(ya.matrix.shape[1]):		
		ya.features = ya.matrix[:,node]
		nuisance_model = LinearRegression()
		nuisance_model.fit(ya.subject_measures[hcpya_remove_linear_vars].values,ya.features) 
		ya.features  = ya.features  - nuisance_model.predict(ya.subject_measures[hcpya_remove_linear_vars].values)
		m = RidgeCV()
		m.fit(ya.features,ya.targets)
		acc.append(pearsonr(m.predict(ya.features),ya.targets)[0])
		models.append(m)

	"""
	apply hcpya to the developmental dataset
	this includes making prediction on a shuffled version of the features (edge weights)
	"""

	r_iters = 100
	predictions = np.zeros((dev.matrix.shape[1],dev.matrix.shape[0]))
	random_predictions = np.zeros((dev.matrix.shape[1],r_iters,dev.matrix.shape[0]))
	for node in range(dev.matrix.shape[1]):
		model = models[node]
		dev.features = dev.matrix[:,node]
		nuisance_model = LinearRegression()
		nuisance_model.fit(dev.subject_measures[remove_linear_vars].values,dev.features) 
		dev.features  = dev.features  - nuisance_model.predict(dev.subject_measures[remove_linear_vars].values)
		predictions[node] = model.predict(dev.features)
		r_feats = dev.features.copy()
		
		for random in range(r_iters):
			np.random.shuffle(r_feats)
			random_predictions[node,random] = model.predict(r_feats)

	prediction_acc = pennlinckit.utils.matrix_corr(predictions[:,mask],dev.subject_measures.bpd_score.values[mask])

	#save the prediction accuracy brain
	mask = np.isnan(dev.subject_measures.bpd_score.values)==False
	np.save('/{0}/data/ridge/hcpya2hcpd_{1}_prediction_acc_{2}.npy'.format(homedir,source,regress_name),prediction_acc)

	#calculate the random prediction accuracy
	random_pred_acc = np.zeros((dev.matrix.shape[1],r_iters))
	for random in range(r_iters):
		random_pred_acc[:,random] = pennlinckit.utils.matrix_corr(random_predictions[:,random,mask],dev.subject_measures.bpd_score.values[mask])

	# how different is random accuracy from the real ones?
	prediction_p = np.zeros(dev.matrix.shape[1])
	for node in range(dev.matrix.shape[1]):
		prediction_p[node] = scipy.stats.ttest_1samp(random_pred_acc[node],prediction_acc[node])[1]


	colors = np.array(pennlinckit.utils.make_heatmap(pennlinckit.utils.cut_data(prediction_acc,1.5),sns.color_palette("light:r", as_cmap=False,n_colors=1001)))
	out_path='/{0}/brains/prediction_acc_all'.format(homedir)
	pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')

	1/0

	correction = multipletests(prediction_p,method='bonferroni')[0]
	correction[prediction_acc<0.0] = False
	sig_pred = prediction_acc.copy()
	colors = np.zeros((400,4))
	colors[correction==True,:3] = np.array(pennlinckit.utils.make_heatmap(pennlinckit.utils.cut_data(sig_pred[correction==True],1.5),sns.color_palette("light:r", as_cmap=False,n_colors=1001)))
	colors[correction==True,3] = 1
	colors[correction==False,3] = 0
	
	out_path='/{0}/brains/prediction_acc_sig'.format(homedir)
	pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')

	# 1/0

	plt.close()
	x,y = predictions[:,mask].mean(axis=0),dev.subject_measures['bpd_score'].values[mask]
	ax = sns.regplot(x=x,y=y)
	plt.ylabel('bdp score')
	plt.xlabel('predicted bdp score')
	plt.tight_layout()
	r,l,h,p = pennlinckit.utils.bootstrap_corr(x,y,pearsonr,1000)
	print (r,l,h,p)
	plt.text(1,.95,'r={0},p={1}\n95%CI:{2},{3}'.format(np.around(r,2),np.around(p,4),np.around(l,2),np.around(h,2)),transform=ax.transAxes,horizontalalignment='right',verticalalignment='top')
	sns.despine()
	plt.savefig('/{0}/figures/hcpya2hcpd_{1}_prediction_whole_brain_{2}.pdf'.format(homedir,source,regress_name))
	plt.close()

	df = pd.DataFrame(columns=['node','accuracy','network'])
	df['node'] = range(400)
	df['accuracy'] = prediction_acc
	df['network'] = pennlinckit.brain.yeo_partition(7)[0]
	order = df.groupby('network').mean().sort_values('accuracy',ascending=True).index.values
	sns.boxenplot(data=df,x='network',y='accuracy',order=order)
	plt.xticks(rotation=90)
	for idx,network in enumerate(order):
		x,y = predictions[pennlinckit.brain.yeo_partition(7)[0]==network].mean(axis=0)[mask],dev.subject_measures['bpd_score'].values[mask]
		r = pearsonr(x,y)[0]
		plt.plot(idx,r,marker="D")

	plt.tight_layout()
	sns.despine()
	plt.savefig('/{0}/figures/hcpya2hcpd_{1}_prediction_networks_{2}.pdf'.format(homedir,source,regress_name))
	plt.close()
	# 1/0

	"""
	let's see if we can reverse engineer the bpd 
	metric by correlating the neo questions with the hcpya prediction
	"""
	mean_prediction = predictions.mean(axis=0)
	reverse = [40,5,50,31,8,33]
	regular = [14,39,54,9,59,24,29,45,55,22,36,21,26,41,51,11,28,30]
	neo_cols = []
	rs = []
	for i in range(61):
		n = 'nffi_{0}'.format(i)
		if n in dev.subject_measures.columns.values:
			neo_cols.append(i)
			rs.append(pennlinckit.utils.nan_pearsonr(dev.subject_measures[n],mean_prediction)[0])
	neo = pd.read_csv('/cbica/projects/hcpd/data/hcpd_behavior_files/nffi01.txt',header=[0,1],sep='\t',low_memory=False)  
	bdp_rev_found = np.intersect1d(np.array(neo_cols)[np.argsort(rs)][:5],reverse).shape[0]
	bdp_reg_found = np.intersect1d(np.array(neo_cols)[np.argsort(rs)][-18:],regular).shape[0]
	print ('{0} regular bpd items found'.format(bdp_reg_found))
	print ('{0} reverse bpd items found'.format(bdp_rev_found))
	
	print ('regular score')
	for found in np.intersect1d(np.array(neo_cols)[np.argsort(rs)][-18:],regular):
		found = 'nffi_' + str(found)
		print (neo[found].columns[0])	
	print ('reverse score')
	for found in np.intersect1d(np.array(neo_cols)[np.argsort(rs)][:5],reverse):
		found = 'nffi_' + str(found)
		print (neo[found].columns[0])	
	

	# null model for finding neo items
	reg = []
	rev = []
	null_mean_prediction = predictions.mean(axis=0)
	for i in range(1000):
		np.random.shuffle(null_mean_prediction)
		neo_cols = []
		rs = []
		for i in range(61):
			n = 'nffi_{0}'.format(i)
			if n in dev.subject_measures.columns.values:
				neo_cols.append(i)
				rs.append(pennlinckit.utils.nan_pearsonr(dev.subject_measures[n],null_mean_prediction)[0])
		rev.append(np.intersect1d(np.array(neo_cols)[np.argsort(rs)][:5],reverse).shape)
		reg.append(np.intersect1d(np.array(neo_cols)[np.argsort(rs)][-18:],regular).shape)

	print ('null model for neo-bpd finder')
	print (scipy.stats.ttest_1samp(reg,bdp_reg_found))
	print (scipy.stats.ttest_1samp(rev,bdp_rev_found))

	for m in dev.subject_measures.columns.values: 
		try:
			assert len(dev.subject_measures[m].values[np.isnan(dev.subject_measures[m].values)==False]) > 100
			r = pennlinckit.utils.nan_pearsonr(dev.subject_measures[m].values,mean_prediction)[0]
			if abs(r) > .17 : print (m,r)
		except:continue

	eat_names = []
	eat_array = []

	for m in dev.subject_measures.columns.values: 
		if m[:4] =='eatq':
			if m[-12:] == 'nm_caregiver': continue
			if m[-7:] == 'nm_self': continue
			print (m)
			eat_names.append(m)
			a = dev.subject_measures[m].copy()
			a[np.isnan(a)==False] = scipy.stats.zscore(a[np.isnan(a)==False])
			eat_array.append(a)
	eat_array = np.array(eat_array)	
	qs = []
	rs = []
	reverse = []
	mean_prediction = predictions.mean(axis=0)
	for i in range(100000):
		choice = np.random.choice(range(eat_array.shape[0]),20,replace=False)
		qs.append(choice)
		rs.append(pennlinckit.utils.nan_pearsonr(np.nanmean(eat_array[choice],axis=0),mean_prediction)[0])
	eatq= pd.read_csv('/cbica/projects/hcpd/data/hcpd_behavior_files/eatq01.txt',header=[0,1],sep='\t',low_memory=False)  
	for found in np.array(eat_names)[qs[np.argmax(rs)].astype(int)]:
		print (found)
		found = '_'.join(found.split('_')[:-1])
		print (eatq[found].columns[0])

def matrix_corr(source):
	data = pennlinckit.utils.load_dataset('/{0}/data/{1}.matrices'.format(homedir,source))


	scores = np.zeros((400,400))
	for i in range(400):
		scores[i] = pennlinckit.utils.matrix_corr(data.matrix[:,i].swapaxes(0,1),data.subject_measures.bpd_score.values)

	labels = pennlinckit.brain.yeo_partition(7)[0]
	label_mask = np.linspace(0,399,10).astype(int)
	sns.heatmap(scores,vmin=-.1,vmax=.1)
	plt.yticks(np.linspace(0,399,10),labels[label_mask])
	plt.xticks(np.linspace(0,399,10),labels[label_mask])
	plt.tight_layout()
	plt.savefig('matrixcorr.pdf')

def region_predict(data,node,**model_args):
	data.targets = data.subject_measures['bpd_score'].values
	data.features = data.matrix[:,node]
	model_args['self'] = data
	pennlinckit.utils.predict(**model_args)

def predict(node,source,age,sex):
	regressors = ['meanFD']
	if age == 1: regressors.append('interview_age')
	if sex == 1: regressors.append('gender_dummy')
	print (regressors)
	
	data = pennlinckit.utils.load_dataset('/{0}/data/{1}.data'.format(homedir,source))
	data.filter(way='has_subject_measure',value='meanFD')

	gender = np.zeros((data.subject_measures.shape[0]))
	if source == 'hcpya': gender[data.subject_measures.Gender=='F'] = 1
	if source == 'hcpd-dcan': gender[data.subject_measures.sex=='F'] = 1
	data.subject_measures['gender_dummy'] = gender

	data.filter(way='has_subject_measure',value='bpd_score')
	data.filter('<',.2,'meanFD')
	# region_predict(data,node,**{'model':'deep','cv':'KFold','folds':5,'neurons':400,'layers':10,'remove_linear_vars':['gender_dummy','motion']})
	region_predict(data,node,**{'model':'ridge','cv':'KFold','folds':10,'remove_linear_vars':regressors})
	np.save('{0}/data/ridge/{1}_prediction_{2}_{3}.npy'.format(homedir,source,node,'_'.join(regressors)),data.prediction)

def submit_predict(source,age,sex):
	"""
	The above function makes the predictions for each factor
	this submit it
	"""
	for node in range(400):
		script_path = '/cbica/home/bertolem/bpd/bpd.py predict {0} {1} {2} {3}'.format(node,source,age,sex) #it me
		pennlinckit.utils.submit_job(script_path,'p{0}'.format(node),RAM=12,threads=1)

def analyze_predict(source,age,sex):
	# %matplotlib inline
	regressors = ['meanFD']
	if age == 1: regressors.append('interview_age')
	if sex == 1: regressors.append('gender_dummy')
	print (regressors)

	data = pennlinckit.utils.load_dataset('/{0}/data/{1}.data'.format(homedir,source))


	if source == 'hcpya': data.subject_measures = data.subject_measures.rename(columns={'Gender': 'sex'})
	data.filter(way='has_subject_measure',value='bpd_score')
	data.filter('<',.2,"meanFD")
	data.filter(way='has_subject_measure',value='meanFD')

	# scipy.stats.ttest_ind(data.subject_measures.bpd_score[data.subject_measures.Gender=='M'],data.subject_measures.bpd_score[data.subject_measures.Gender=='F'])
	sns.displot(data=data.subject_measures, x="bpd_score", hue="sex", kind="kde")
	sns.despine()
	plt.savefig('/{0}/figures/{1}_gender.pdf'.format(homedir,source))
	plt.close()


	prediction = np.zeros((400,data.subject_measures.shape[0]))
	for node in range(400):
		prediction[node] = np.load('{0}data/ridge/{1}_prediction_{2}_{3}.npy'.format(homedir,source,node,'_'.join(regressors)))

	prediction_acc = pennlinckit.utils.matrix_corr(prediction,data.subject_measures.bpd_score.values)

	plt.close()
	x,y = prediction.mean(axis=0),data.subject_measures['bpd_score'].values
	ax = sns.regplot(x=x,y=y)
	plt.ylabel('bdp score')
	plt.xlabel('predicted bdp score')
	plt.tight_layout()
	r,l,h,p = pennlinckit.utils.bootstrap_corr(x,y,pearsonr,1000)
	print (r,l,h,p)
	plt.text(1,.95,'r={0},p={1}\n95%CI:{2},{3}'.format(np.around(r,2),np.around(p,4),np.around(l,2),np.around(h,2)),transform=ax.transAxes,horizontalalignment='right',verticalalignment='top')
	sns.despine()
	plt.savefig('/{0}/figures/{1}_{2}_prediction_whole_brain.pdf'.format(homedir,source,'_'.join(regressors)))
	plt.close()

	df = pd.DataFrame(columns=['node','accuracy','network'])
	df['node'] = range(400)
	df['accuracy'] = prediction_acc
	df['network'] = pennlinckit.brain.yeo_partition(7)[0]
	order = df.groupby('network').mean().sort_values('accuracy',ascending=True).index.values
	sns.boxenplot(data=df,x='network',y='accuracy',order=order)
	plt.xticks(rotation=90)
	for idx,network in enumerate(order):
		x,y = prediction[pennlinckit.brain.yeo_partition(7)[0]==network].mean(axis=0),data.subject_measures['bpd_score'].values
		r = pearsonr(x,y)[0]
		plt.plot(idx,r,marker="D")

	plt.tight_layout()
	sns.despine()
	plt.savefig('/{0}/figures/{1}_{2}_prediction_network.pdf'.format(homedir,source,'_'.join(regressors)))
	plt.close()

	# colors = np.array(pennlinckit.utils.make_heatmap(pennlinckit.utils.cut_data(prediction_acc,1.5),sns.diverging_palette(220, 10,n=1001)))
	# out_path='/{0}/brains/{1}_{2}_prediction_acc'.format(homedir,source,'_'.join(regressors))
	# pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')

def developmental(node):
	data = load_data('hcpd-dcan')
	node_coefs = np.zeros((400))
	for node2 in range(400):
		if node2 == node:continue
		node_df = pd.DataFrame(columns=['fc','age','bpd'])
		node_df['fc'] = data.matrix[:,node,node2]
		node_df['bpd'] = data.subject_measures.bpd_score.values
		node_df['age'] = data.subject_measures.interview_age.values
		#does bpd increase as fc and age increase together?
		model = sm.OLS.from_formula(formula='bpd ~ fc + age + age:fc', data=node_df).fit()
		result_df = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
		node_coefs[node2] = result_df['coef'][-1]
	np.save('{0}/data/dev/{1}_age_fc_interaction.npy'.format(homedir,node),node_coefs)

def submit_developmental():
	"""
	The above function makes the predictions for each factor
	this submit it
	"""
	for node in range(400):
		script_path = '/cbica/home/bertolem/bpd/bpd.py developmental {0}'.format(node) #it me
		pennlinckit.utils.submit_job(script_path,'d_{0}'.format(node),RAM=12,threads=1)

def dev_region_predict(data,node,**model_args):
	data.targets = data.subject_measures['interview_age'].values
	data.features = data.matrix[:,node]
	model_args['self'] = data
	pennlinckit.utils.predict(**model_args)

def dev_predict(node):
	regressors = ['meanFD']
	data = load_data('hcpd-dcan',['meanFD'])
	dev_region_predict(data,node,**{'model':'ridge','cv':'KFold','folds':10,'remove_linear_vars':regressors})
	np.save('{0}/data/ridge/age_prediction_{1}_{2}.npy'.format(homedir,node,'_'.join(regressors)),data.prediction)

def submit_dev_predict():
	"""
	The above function makes the predictions for each factor
	this submit it
	"""
	for node in range(400):
		script_path = '/cbica/home/bertolem/bpd/bpd.py dev_predict {0}'.format(node) #it me
		pennlinckit.utils.submit_job(script_path,'p{0}'.format(node),RAM=12,threads=1)

def analyze_developmental():
	data = load_data('hcpd-dcan',['meanFD'])
	age_acc = np.zeros((400))
	for node in range(400):
		age_acc[node] = pearsonr(data.subject_measures.interview_age,np.load('{0}/data/ridge/age_prediction_{1}_{2}.npy'.format(homedir,node,'_'.join(['meanFD']))))[0]

	p_acc = np.load('/{0}/data/ridge/hcpya2hcpd_full_prediction_acc_meanFD.npy'.format(homedir))

	real_r = pearsonr(p_acc,age_acc)[0]
	spin = pennlinckit.brain.spin_test(p_acc,age_acc)
	print (pennlinckit.brain.spin_stat(p_acc,age_acc,spin))

	sns.displot(spin)
	plt.vlines(real_r,0,100)
	plt.savefig('{0}/figures/dev_spin.pdf'.format(homedir))

	colors = np.array(pennlinckit.utils.make_heatmap(pennlinckit.utils.cut_data(age_acc,1.5),sns.color_palette("light:r", as_cmap=False,n_colors=1001)))
	out_path='/{0}/brains/prediction_age'.format(homedir)
	pennlinckit.brain.write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command')


if len(sys.argv) > 1:
	if sys.argv[1] == 'make_data': make_data(sys.argv[2])
	if sys.argv[1] == 'predict': 
		predict(int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),int(sys.argv[5]))
	if sys.argv[1] == 'dev_predict': 
		dev_predict(int(sys.argv[2]))
	if sys.argv[1] == 'developmental': 
		developmental(int(sys.argv[2]))