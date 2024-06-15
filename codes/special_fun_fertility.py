import pandas as pd
import numpy as np
import pickle
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import statsmodels.api as sm
from scipy import stats
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KernelDensity
import scipy.stats as st
import scipy.special as sp


plot_folder='../Figures/'
data_folder='../data/'


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def preprocessing(manifests_df,count_df):
    ''' after genrating count matrix we need to prunn headers and remove some of the noise '''
    ########### input parameters
    #  1) manifests_df : prepared by claire (NGI Sample ID	Sample name	Sample description	GCKO2	145480	pr	t	d	b	mf	dr)
    #  2) count file came by barseq pipeline
    #############################

    ########### output parameters
    # cmd_df  : this is the dataframe where final_count_matrix is combined with manifests dataframe
    # final_df: we added forward and reverse reads in this data file
    # final_df_two_read: here forward and reverse reads are sperated
    # manifests_df
    #############################


    # STEP1: We want to remove for one read we have two columns.
    tmp_df=count_df.loc[:,count_df.columns[2:]].copy()

    tmp_df.loc['col_sum',:]=tmp_df.sum(axis=0)
    # tmp_df.loc[:,'row_sum']=tmp_df.sum(axis=1)

    # these are the required columns which we will use in analysis
    req_cols=tmp_df.columns[tmp_df.loc['col_sum']>20].to_list()  ####



    ### for smotth filtering for rows
    tmp_df=tmp_df/tmp_df.loc['col_sum',:]
    bool_df=(tmp_df<1e-5)
    req_rows=bool_df.index[~bool_df.all(axis=1)].to_list()
    req_rows.remove('col_sum')

    ##### total colums should be in our dataFrame
    cols=['Gene',  'Barcodes']
    for col in req_cols:
        cols.append(col)

    ### this is our first dataFrame comes after deleting garbage
    req_df=count_df.loc[req_rows,cols].copy()
    req_df['pbanka_id']=[item[0] for item in req_df['Gene'].str.findall('PBANKA_[0-9]*')]


    ## this code is ngi to sample

    final_df=pd.DataFrame(index=req_df.index,columns=['Gene',  'Barcodes','pbanka_id'])
    final_df.loc[:,['Gene',  'Barcodes','pbanka_id']]= req_df.loc[:,['Gene',  'Barcodes','pbanka_id']].copy()
    final_df_des=final_df.copy()
    final_df_two_read=pd.DataFrame(index=req_df.index,columns=['Gene',  'Barcodes','pbanka_id'])
    final_df_two_read.loc[:,['Gene',  'Barcodes','pbanka_id']]= req_df.loc[:,['Gene',  'Barcodes','pbanka_id']].copy()
    sample_to_read={} ### this is the dictionary where we will map the reads
    manifests_df.set_index('NGI Sample ID',inplace=True)
    manifests_df=manifests_df.fillna('NA')
    for ngi in manifests_df.index:

       ### get two reads
       sam= manifests_df.loc[ngi,'Sample description'] # sample description
       two_reads=req_df.columns[req_df.columns.str.contains(ngi)]
       if len(two_reads)==2:
           # final_df[ngi]=np.nan()
           final_df.loc[:,ngi]=req_df.loc[:,two_reads].mean(axis=1)
           final_df_des.loc[:,sam]=req_df.loc[:,two_reads].mean(axis=1)
           # final_df_two_read[ngi+'.read1']=np.nan()
           # final_df_two_read[ngi+'.read2']=np.nan()
           final_df_two_read.loc[:,sam+'.read1']=req_df.loc[:,two_reads[0]]
           final_df_two_read.loc[:,sam+'.read2']=req_df.loc[:,two_reads[1]]
       elif len(two_reads)==1:
           # final_df[ngi]=np.nan()
           final_df.loc[:,ngi]=req_df.loc[:,two_reads].copy()
           final_df_des.loc[:,sam]=req_df.loc[:,two_reads].copy()
           print('Please check there is only one reads for the sample: %s'%ngi)
       else:

           print('Number of reads are mismatched for the sample: %s'%ngi)

    return final_df,final_df_des,final_df_two_read,manifests_df




def filter_input_dropout(df,df_s,df_read,input_df,manfest_df,percent=0.9,rel_cut=1e-5,day0='d0'):
    ''' We are going to find dropouts as well as how many we can map to inputs'''
    ########## input parameters
    # df: original count dataframe (read1+ read2)
    # df_read is dtaframe when reads are sperated
    # input_df: This is the DATAFRAME  which one used as a input for pools
    # manfest_df: manifests dtaframe
    ########

    ########## output parameters
    ## filtered_count_df
    ## filtered_df_read
    ########

    ## we will drop sum string columns and work on numerical values
    tmp=df.copy()
    tmp=tmp.drop(columns=['Gene','Barcodes','pbanka_id'])
    tmp=tmp.div(tmp.sum(axis=0))

    cols=[]

    for k,v in manfest_df.groupby('t').indices.items():
        if not (k =='NA'):
            for id in v:
                cols.append(manfest_df.index[id])
    ### these are the columns

    test1=tmp[cols].copy()
    filtered_count_df=df[(test1<rel_cut).mean(axis=1)<percent]
    filtered_df_read=df_read[(test1<rel_cut).mean(axis=1)<percent]

    drop_out_genes=set(input_df['gene_id'])-set(filtered_count_df['pbanka_id'])

    print('Number of Dropout genes at vitro level : %d' %len(drop_out_genes))
    print('Genes are :',drop_out_genes)

    # now we are going to check dropout gene at day 0 level

    day_cols=[]

    for k,v in manfest_df.groupby('d').indices.items():
        if k ==day0:
            for id in v:
                day_cols.append(manfest_df.index[id])

    #####

    tmp=filtered_count_df.copy()
    tmp=tmp.drop(columns=['Gene','Barcodes','pbanka_id'])
    tmp=tmp.div(tmp.sum(axis=0))
    test1=tmp[day_cols].copy()

    filtered_count_df=filtered_count_df[(test1<rel_cut).mean(axis=1)<percent]
    filtered_df_read=filtered_df_read[(test1<rel_cut).mean(axis=1)<percent]

    drop_out_genes=set(input_df['gene_id'])-set(filtered_count_df['pbanka_id'])

    print('Number of Dropout genes at d0 level : %d' %len(drop_out_genes))
    print('Genes are :',drop_out_genes)

    print('Number of final genes : %d' %len(set(input_df['gene_id'])&set(filtered_count_df['pbanka_id'])))
    contaminated_genes=list(set(filtered_count_df['pbanka_id'])-set(input_df['gene_id']))
    print('contaminated genes  :' ,contaminated_genes)

    ### we are going to drop contaminated genes
    filtered_count_df.set_index('pbanka_id',inplace=True)
    filtered_df_read.set_index('pbanka_id',inplace=True)

    filtered_count_df=filtered_count_df.drop(contaminated_genes)
    filtered_df_read=filtered_df_read.drop(contaminated_genes)

    df_s.set_index('pbanka_id',inplace=True)
    filltered_df_s=df_s.loc[filtered_count_df.index,:].copy()

    return filtered_count_df,filtered_df_read,filltered_df_s


def relative_abundance_analysis(df,manfest_df,prev_to_new,db_df,plot_info=None):
    ''' We are going to do relative abundance analysis.
    With this analysis we will plot rlative abundance of each gene over time
    '''

    ########## input parameters ##############
    #### df: dataframe of count matrix
    ## manfest_df : manifest dataframe
    ##prev_to_new: previous pbnaka id to new pbanka id
    ##db_df : gene name and description
    # plot_info is not none then we would like to pass some variables for plot_info
    ##########  ##############
     ### this trick we used to remove zero count
    rel_df=df.copy()
    rel_df=rel_df.drop(columns=['Gene','Barcodes'])
    rel_df= rel_df + 1
    rel_df=rel_df.div(rel_df.sum(axis=0), axis=1)
    rel_df=np.log2(rel_df)

    ### convert old pbanka to new ids
    geneConv,old_to_new_ids,geneConv_new=getNewIdfromPrevID(rel_df.index,prev_to_new,db_df)
    rel_df=rel_df.rename(old_to_new_ids, axis='index')

    ### find duplicates
    if not(len(rel_df.index.unique())==rel_df.shape[0]):
        print('some gene are duplicate. check function relative_abundance_analysis')
        exit()

    ## group by based on mossi feed 1 and 2
    ###### this is for the female

    tmp_mn=pd.DataFrame(index=rel_df.index)
    tmp_sd=pd.DataFrame(index=rel_df.index)
    for k,v in manfest_df.groupby(['sex','d','mf']).indices.items():
        if (not(k[1]=='NA')) and (not(k[2]=='NA')):
            key=k[0]+'_'+k[1]+'_'+k[2]
            tmp_mn[key]=rel_df[manfest_df.index[v]].mean(axis=1).copy()
            tmp_sd[key]=rel_df[manfest_df.index[v]].std(axis=1).copy()
    sex_list=[tmp_mn,tmp_sd]## this is the list of dataframe
    ###### this is for the male



    #### we would like to plot on pdf file
    import pdb; pdb.set_trace()
    if not (plot_info==None):
        # import pdb;pdb.set_trace()
        plot_relative_abunndance_daywise(sex_list,geneConv_new,plot_info)


def calculate_mean_sd_groupby_pcr(df,manfest_df,grp_cols,day_pos,days=['d0','d13']):

    ''' We are going to calculate mean and SD of slected columns '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_sd=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)

    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)
            tmp_mn[key]=df[manfest_df.index[v]].mean(axis=1).copy()

            tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])
            tmp_mn_log[key]=df_log[manfest_df.index[v]].mean(axis=1).copy()

    listofvar=[]
    for day in days:
        day0_cols=tmp_mn.columns[tmp_mn.columns.str.contains(day)]
        d0_df_mean,d0_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day0_cols,day=day)
        listofvar.append(d0_df_mean.copy())
        listofvar.append(d0_df_sd.copy())

    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return listofvar


def calculate_mean_sd_groupby_tube(df,manfest_df,grp_cols,day_pos,days=['NA']):

    ''' We are going to calculate mean and SD of slected columns '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_CV=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('t')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()
            for dr,dr_v in tmp_mani.groupby('t').indices.items():

                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                ### check whether there is zero variance
                # tmp_var[tmp_var<1e-10]=1e-10;
                tmp_mn_log[key+'_'+dr]=df_log[tmp_mani.index[dr_v]].mean(axis=1).copy()
                df_sample[key+'_'+dr]=tmp_mani.index[dr_v].shape[0]

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=np.log2(mean_df.copy())
            final_CV[key]=np.square(sd_max.copy()/mean_df.copy())


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])



    listofvar=[]
    for day in days:
        day0_cols=final_mean.columns[final_mean.columns.str.contains(day)]
        d0_df_mean,d0_df_sd=multi_to_one_df(final_mean,final_CV,day0_cols,day=day)
        listofvar.append(d0_df_mean.copy())
        listofvar.append(d0_df_sd.copy())

    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return listofvar



def propagate_error_cuvette(df,manfest_df,grp_cols,day_pos,days=['NA']):

    ''' We are going to calculate mean and SD of slected columns '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_CV=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('t')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()
            for dr,dr_v in tmp_mani.groupby('t').indices.items():

                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                ### check whether there is zero variance
                # tmp_var[tmp_var<1e-10]=1e-10;
                tmp_mn_log[key+'_'+dr]=df_log[tmp_mani.index[dr_v]].mean(axis=1).copy()
                df_sample[key+'_'+dr]=tmp_mani.index[dr_v].shape[0]

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)

            final_mean[key]=mean_df.copy()
            final_CV[key]=var_max.copy()


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])

    final_mean_log=np.log2(final_mean)
    final_var_log=(final_CV)/((final_mean*final_mean)*((np.log(10)**2)*np.log10(2)))



    return final_mean_log,final_var_log

def weighted_mean_variance(rel_fit,rel_err):

    ''' With this analysis we would like to analyze weigted varaiace analysis'''

    #### input parameters
    # rel_fit: dataframe with two cols  mean
    # rel_err: dataframe with two cols variance
    ##
    weight = rel_err.rdiv(1)

    nume_df=pd.DataFrame(rel_fit.values * weight.values, columns=rel_fit.columns, index=rel_fit.index)
    nume= nume_df.sum(axis=1)
    deno=weight.sum(axis=1)
    # calculate mean
    mean_df=nume/deno

    # calculate first varaiance
    var1= weight.sum(axis=1).rdiv(1)

    # claculate second variance
    term1=rel_fit.iloc[:, 0:].sub(mean_df, axis=0).pow(2,axis=1)
    term3=term1.div(rel_err,axis=0) ### divided by variance
    term4=term3.sum(axis=1)
    term2=(1/(rel_fit.shape[1]-1))


    var2=var1*term4*term2

    var_max=pd.concat([var1, var2], axis=1).max(axis=1)
    sd_max=var_max.apply(np.sqrt)

    return [mean_df,sd_max,var_max]

def calculate_mean_sd_groupby_dissection(df,manfest_df,grp_cols,day_pos,days=['d0','d13']):

    ''' We are going to calculate mean and SD of slected columns '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_CV=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('dr')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()

            for dr,dr_v in tmp_mani.groupby('dr').indices.items():

                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                ### check whether there is zero variance
                # tmp_var[tmp_var<1e-10]=1e-10;
                tmp_mn_log[key+'_'+dr]=df_log[tmp_mani.index[dr_v]].mean(axis=1).copy()
                df_sample[key+'_'+dr]=tmp_mani.index[dr_v].shape[0]

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=np.log2(mean_df.copy())
            final_CV[key]=np.square(sd_max.copy()/mean_df.copy())


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])



    listofvar=[]
    for day in days:
        day0_cols=final_mean.columns[final_mean.columns.str.contains(day)]
        d0_df_mean,d0_df_sd=multi_to_one_df(final_mean,final_CV,day0_cols,day=day)
        listofvar.append(d0_df_mean.copy())
        listofvar.append(d0_df_sd.copy())

    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return listofvar


def calculate_mean_sd_groupby_mossifeed(df,manfest_df,grp_cols,day_pos,days=['d0']):

    ''' We are going to calculate mean and SD of slected columns '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_CV=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('mf')

    grp_cols.remove('b')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()

            for dr,dr_v in tmp_mani.groupby([ 'b','mf']).indices.items():

                dr='_'.join(dr)
                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                ### check whether there is zero variance
                # tmp_var[tmp_var<1e-10]=1e-10;
                tmp_mn_log[key+'_'+dr]=df_log[tmp_mani.index[dr_v]].mean(axis=1).copy()
                df_sample[key+'_'+dr]=tmp_mani.index[dr_v].shape[0]

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=np.log2(mean_df.copy())
            final_CV[key]=np.square(sd_max.copy()/mean_df.copy())


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])



    listofvar=[]
    for day in days:
        day0_cols=final_mean.columns[final_mean.columns.str.contains(day)]
        d0_df_mean,d0_df_sd=multi_to_one_df(final_mean,final_CV,day0_cols,day=day)
        listofvar.append(d0_df_mean.copy())
        listofvar.append(d0_df_sd.copy())

    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return listofvar



def seperate_mossifeed_day0_mean_and_CV(df,manfest_df,grp_cols,day_pos,days=['d0']):

    ''' We are going to calculate mean and SD of slected columns '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_CV=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('mf')

    grp_cols.remove('b')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()

            for dr,dr_v in tmp_mani.groupby([ 'b','mf']).indices.items():

                dr='_'.join(dr)
                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                ### check whether there is zero variance
                # tmp_var[tmp_var<1e-10]=1e-10;
                tmp_mn_log[key+'_'+dr]=df_log[tmp_mani.index[dr_v]].mean(axis=1).copy()
                df_sample[key+'_'+dr]=tmp_mani.index[dr_v].shape[0]

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=np.log2(mean_df.copy())
            final_CV[key]=np.square(sd_max.copy()/mean_df.copy())


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])



    listofvar=[]
    for day in days:
        day0_cols=final_mean.columns[final_mean.columns.str.contains(day)]
        d0_df_mean,d0_df_sd=multi_to_one_df(final_mean,final_CV,day0_cols,day=day)
        listofvar.append(d0_df_mean.copy())
        listofvar.append(d0_df_sd.copy())

    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return listofvar



def calculate_mean_sd_groupby_mossifeed_day13(df,manfest_df,grp_cols,day_pos,days=['d13']):

    ''' We are going to calculate mean and SD of slected columns '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_CV=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('mf')

    grp_cols.remove('dr')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()

            for dr,dr_v in tmp_mani.groupby(['dr','mf']).indices.items():

                dr='_'.join(dr)
                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                ### check whether there is zero variance
                # tmp_var[tmp_var<1e-10]=1e-10;
                tmp_mn_log[key+'_'+dr]=df_log[tmp_mani.index[dr_v]].mean(axis=1).copy()
                df_sample[key+'_'+dr]=tmp_mani.index[dr_v].shape[0]

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=np.log2(mean_df.copy())
            final_CV[key]=np.square(sd_max.copy()/mean_df.copy())


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])



    listofvar=[]
    for day in days:
        day0_cols=final_mean.columns[final_mean.columns.str.contains(day)]
        d0_df_mean,d0_df_sd=multi_to_one_df(final_mean,final_CV,day0_cols,day=day)
        listofvar.append(d0_df_mean.copy())
        listofvar.append(d0_df_sd.copy())

    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return listofvar

def getCombined_mean_variance(df_m,df_var,df_sample):
    ''' We are going to combine mean and variance '''

    ### input parameters
    # df_m  mean dataframes  #### df  col1 col2 col3
    # df_var  std_dataframes
    # df_sample: number of samples for each column

    #### end of parameters

    # ### calculate combined mean
    # nume_df=pd.DataFrame(df_m.values * df_sample.values, columns=df_m.columns, index=df_m.index)
    # numerator=nume_df.sum(axis=1).copy()
    # mean_df=numerator/df_sample.sum(axis=1)['num']
    #
    # ### calculate combined variance
    #
    # term1=df_m.iloc[:, 0:].sub(mean_df, axis=0).pow(2,axis=1)  ## (x-Xavg)*(x-Xavg)
    # term2=df_var+term1
    # nume_df=pd.DataFrame(term2.values * df_sample.values, columns=df_m.columns, index=df_m.index)
    # numerator=nume_df.sum(axis=1).copy()
    # var_df=numerator/df_sample.sum(axis=1)['num']
    # sd_df=var_df.apply(np.sqrt)

    mean_df=df_m.mean(axis=1)
    sum_df=df_var.sum(axis=1)
    var_df=sum_df.copy()/np.square(df_var.shape[1])
    sd_df=var_df.apply(np.sqrt)

    return [mean_df,sd_df,var_df]





def multi_to_one_df(tmp_mn,tmp_sd,day_cols,day='d0'):
    ''' convert a columns dataframe from multiple columns '''
    ##### input parameters ###
    # tmp_mn= mean dataframe
    # tmp_sd= sd dataframe
    # day_cols=columns to merge
    ####
    tmp_all_mean=pd.DataFrame()
    tmp_all_sd=pd.DataFrame()
    list_df_mn=[] ### this is the list pf dataframes for mean
    list_df_sd=[] ### this is the list pf dataframes for std

    for col in day_cols:
        idx=tmp_mn.index+'_'+col ### these are the new index
        tmp_all_mean['index']=idx
        tmp_all_sd['index']=idx
        ## fill mean
        arr=tmp_mn[col].to_numpy()
        tmp_all_mean[day+'_mean']=arr
        ## fill std
        arr=tmp_sd[col].to_numpy()
        tmp_all_sd[day+'_std']=arr
        list_df_mn.append(tmp_all_mean.copy())
        list_df_sd.append(tmp_all_sd.copy())

    # concat all dataframe
    final_df_mean=pd.concat(list_df_mn)
    final_df_std=pd.concat(list_df_sd)

    return final_df_mean,final_df_std


def calculate_mean_sd_groupby_blood(df,manfest_df,grp_cols,day_pos,days=['d0','d13']):

    ''' We are going to calculate mean and SD of slected columns '''

    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_CV=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('b')



    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()
            for dr,dr_v in tmp_mani.groupby('b').indices.items():

                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                ### check whether there is zero variance
                # tmp_var[tmp_var<1e-10]=1e-10;
                tmp_mn_log[key+'_'+dr]=df_log[tmp_mani.index[dr_v]].mean(axis=1).copy()
                df_sample[key+'_'+dr]=tmp_mani.index[dr_v].shape[0]
            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=np.log2(mean_df.copy())
            final_CV[key]=np.square(sd_max.copy()/mean_df.copy())


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])



    listofvar=[]
    for day in days:
        day0_cols=final_mean.columns[final_mean.columns.str.contains(day)]
        d0_df_mean,d0_df_sd=multi_to_one_df(final_mean,final_CV,day0_cols,day=day)
        listofvar.append(d0_df_mean.copy())
        listofvar.append(d0_df_sd.copy())

    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return listofvar



def propagate_error_gDNA_extraction_method(df,manfest_df,grp_cols,day_pos,days=['d0']):

    ''' We are going to calculate mean and SD of slected columns '''

    # df_log=np.log2(df)

    # tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_CV=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    # vars=manfest_df['dr'].unique().tolist()
    # vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('e')



    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()
            for dr,dr_v in tmp_mani.groupby('e').indices.items():
                tmp_mn=pd.DataFrame(index=df.index)
                tmp_var=pd.DataFrame(index=df.index)
                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                ### check whether there is zero variance
                # tmp_var[tmp_var<1e-10]=1e-10;
                # tmp_mn_log[key+'_'+dr]=df_log[tmp_mani.index[dr_v]].mean(axis=1).copy()
                # df_sample[key+'_'+dr]=tmp_mani.index[dr_v].shape[0]
                # [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

                # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
                final_mean[key+'_'+dr]=tmp_mn
                final_CV[key+'_'+dr]=np.square(tmp_var.copy()/tmp_mn.copy())
    final_mean_log=np.log2(final_mean)
    final_var_log=(final_CV)/((final_mean*final_mean)*((np.log(10)**2)*np.log10(2)))

    return final_mean_log, final_var_log



def propagate_error_day0(df,manfest_df,grp_cols,day_pos,days=['d0']):

    ''' We are going to calculate mean and SD for combined analyis from PCR to mosquitofeed '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_var=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('mf')

    grp_cols.remove('b')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()

            for dr,dr_v in tmp_mani.groupby([ 'b','mf']).indices.items():

                dr='_'.join(dr)
                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()


            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=mean_df.copy()
            final_var[key]=var_max.copy()


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])





    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return final_mean,final_var


def propagate_error_day0_each_mossifeed(df,manfest_df,grp_cols,day_pos,days=['d0']):

    ''' We are going to calculate mean and SD for combined analyis from PCR to mosquitofeed '''
    df_log=np.log2(df)


    ###
    final_mean_mf1=pd.DataFrame(index=df.index)
    final_var_mf1=pd.DataFrame(index=df.index)
    final_mean_mf2=pd.DataFrame(index=df.index)
    final_var_mf2=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('mf')

    grp_cols.remove('b')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()
            tmp_mn_mf1=pd.DataFrame(index=df.index)
            tmp_var_mf1=pd.DataFrame(index=df.index)

            tmp_mn_mf2=pd.DataFrame(index=df.index)
            tmp_var_mf2=pd.DataFrame(index=df.index)
            for dr,dr_v in tmp_mani.groupby([ 'b','mf']).indices.items():
                if dr[1]=='mf1':
                    dr_j='_'.join(dr)
                    tmp_mn_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                    if df[tmp_mani.index[dr_v]].shape[1]==1:
                        tmp_var_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1,ddof=0).copy()
                    else:
                        tmp_var_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                elif dr[1]=='mf2':
                    dr_j='_'.join(dr)
                    tmp_mn_mf2[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                    tmp_var_mf2[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                else:
                    continue

            # for dr,dr_v in tmp_mani.groupby([ 'mf']).indices.items():
            #
            #     if dr=='mf1':
            #         dr_j='_'.join(dr)
            #         tmp_mn_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
            #         if df[tmp_mani.index[dr_v]].shape[1]==1:
            #             tmp_var_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1,ddof=0).copy()
            #         else:
            #             tmp_var_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
            #     elif dr=='mf2':
            #         dr_j='_'.join(dr)
            #         tmp_mn_mf2[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
            #         tmp_var_mf2[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
            #     else:
            #         continue

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn_mf1,tmp_var_mf1,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean_mf1[key]=mean_df.copy()
            final_var_mf1[key]=var_max.copy()

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn_mf2,tmp_var_mf2,df_sample)
            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean_mf2[key]=mean_df.copy()
            final_var_mf2[key]=var_max.copy()


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])





    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    final_mean_mf1_log=np.log2(final_mean_mf1)
    final_var_mf1_log=(final_var_mf1)/((final_mean_mf1*final_mean_mf1)*((np.log(10)**2)*np.log10(2)))
    final_mean_mf2_log=np.log2(final_mean_mf2)
    final_var_mf2_log=(final_var_mf2)/((final_mean_mf2*final_mean_mf2)*((np.log(10)**2)*np.log10(2)))

    return final_mean_mf1_log,final_var_mf1_log,final_mean_mf2_log,final_var_mf2_log


def multi_to_one_df(tmp_mn,tmp_sd,day_cols,day='d0'):
    ''' convert a columns dataframe from multiple columns '''
    ##### input parameters ###
    # tmp_mn= mean dataframe
    # tmp_sd= sd dataframe
    # day_cols=columns to merge
    ####
    tmp_all_mean=pd.DataFrame()
    tmp_all_sd=pd.DataFrame()
    list_df_mn=[] ### this is the list pf dataframes for mean
    list_df_sd=[] ### this is the list pf dataframes for std

    for col in day_cols:
        idx=tmp_mn.index+'_'+col ### these are the new index
        tmp_all_mean['index']=idx
        tmp_all_sd['index']=idx
        ## fill mean
        arr=tmp_mn[col].to_numpy()
        tmp_all_mean[day+'_mean']=arr
        ## fill std
        arr=tmp_sd[col].to_numpy()
        tmp_all_sd[day+'_std']=arr
        list_df_mn.append(tmp_all_mean.copy())
        list_df_sd.append(tmp_all_sd.copy())

    # concat all dataframe
    final_df_mean=pd.concat(list_df_mn)
    final_df_std=pd.concat(list_df_sd)

    return final_df_mean,final_df_std



def propagate_error_day13_each_mossifeed(df,manfest_df,grp_cols,day_pos,days=['d13']):

    ''' We are going to calculate mean and SD for combined analyis from PCR to mosquitofeed '''
    df_log=np.log2(df)



    ###
    final_mean_mf1=pd.DataFrame(index=df.index)
    final_var_mf1=pd.DataFrame(index=df.index)
    final_mean_mf2=pd.DataFrame(index=df.index)
    final_var_mf2=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('mf')

    grp_cols.remove('dr')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()
            tmp_mn_mf1=pd.DataFrame(index=df.index)
            tmp_var_mf1=pd.DataFrame(index=df.index)
            tmp_mn_mf2=pd.DataFrame(index=df.index)
            tmp_var_mf2=pd.DataFrame(index=df.index)
            for dr,dr_v in tmp_mani.groupby([ 'dr','mf']).indices.items():

                if dr[1]=='mf1':
                    dr_j='_'.join(dr)
                    tmp_mn_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                    tmp_var_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                elif dr[1]=='mf2':
                    dr_j='_'.join(dr)
                    tmp_mn_mf2[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                    tmp_var_mf2[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
                else:
                    continue

            # for dr,dr_v in tmp_mani.groupby([ 'mf']).indices.items():
            #     import pdb; pdb.set_trace()
            #     if dr=='mf1':
            #         dr_j='_'.join(dr)
            #         tmp_mn_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
            #         tmp_var_mf1[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
            #     elif dr=='mf2':
            #         dr_j='_'.join(dr)
            #         tmp_mn_mf2[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
            #         tmp_var_mf2[key+'_'+dr_j]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
            #     else:
            #         continue

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn_mf1,tmp_var_mf1,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean_mf1[key]=mean_df.copy()
            final_var_mf1[key]=var_max.copy()

            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn_mf2,tmp_var_mf2,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean_mf2[key]=mean_df.copy()
            final_var_mf2[key]=var_max.copy()


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])





    final_mean_mf1_log=np.log2(final_mean_mf1)
    final_var_mf1_log=(final_var_mf1)/((final_mean_mf1*final_mean_mf1)*((np.log(10)**2)*np.log10(2)))
    final_mean_mf2_log=np.log2(final_mean_mf2)
    final_var_mf2_log=(final_var_mf2)/((final_mean_mf2*final_mean_mf2)*((np.log(10)**2)*np.log10(2)))

    return final_mean_mf1_log,final_var_mf1_log,final_mean_mf2_log,final_var_mf2_log

def propagate_error_day13(df,manfest_df,grp_cols,day_pos,days=['d13']):

    ''' We are going to calculate mean and SD for combined analyis from PCR to mosquitofeed '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_var=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    vars=manfest_df['dr'].unique().tolist()
    vars.remove('NA')
    ### get columns that contains

    ###

    grp_cols.remove('mf')

    grp_cols.remove('dr')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)

            tmp_mani=manfest_df.iloc[v].copy()

            for dr,dr_v in tmp_mani.groupby([ 'dr','mf']).indices.items():

                dr='_'.join(dr)
                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()


            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=mean_df.copy()
            final_var[key]=var_max.copy()


            # we are going to apply weighted varaiance analysis
            # tmp_sd[key]=np.square((df[manfest_df.index[v]].std(axis=1).copy())/(df[manfest_df.index[v]].mean(axis=1).copy()))
            # print('hello',manfest_df.iloc[v,:])





    # day13_cols=tmp_mn.columns[tmp_mn.columns.str.contains(days[1])]
    # d13_df_mean,d13_df_sd=multi_to_one_df(tmp_mn_log,tmp_sd,day13_cols,day=days[1])

    return final_mean,final_var


def multi_to_one_df(tmp_mn,tmp_sd,day_cols,day='d0'):
    ''' convert a columns dataframe from multiple columns '''
    ##### input parameters ###
    # tmp_mn= mean dataframe
    # tmp_sd= sd dataframe
    # day_cols=columns to merge
    ####
    tmp_all_mean=pd.DataFrame()
    tmp_all_sd=pd.DataFrame()
    list_df_mn=[] ### this is the list pf dataframes for mean
    list_df_sd=[] ### this is the list pf dataframes for std

    for col in day_cols:
        idx=tmp_mn.index+'_'+col ### these are the new index
        tmp_all_mean['index']=idx
        tmp_all_sd['index']=idx
        ## fill mean
        arr=tmp_mn[col].to_numpy()
        tmp_all_mean[day+'_mean']=arr
        ## fill std
        arr=tmp_sd[col].to_numpy()
        tmp_all_sd[day+'_std']=arr
        list_df_mn.append(tmp_all_mean.copy())
        list_df_sd.append(tmp_all_sd.copy())

    # concat all dataframe
    final_df_mean=pd.concat(list_df_mn)
    final_df_std=pd.concat(list_df_sd)

    return final_df_mean,final_df_std






def applyGLM(df_mean,df_sd,day,cut_sd=None,cut_mean=None):
    ''' We would like to fit generalized linear model with linking function'''
    y=df_sd[day+'_std'].to_numpy()
    x=df_mean[day+'_mean'].to_numpy()

    if (not(cut_sd==None)) and (not(cut_mean==None)):
        idx_mean = np.where(x>cut_mean)[0]
        idx_sd = np.where(y>cut_sd)[0]
        idx=np.intersect1d(idx_mean,idx_sd)
        y,x = y[idx],x[idx]
    exog, endog = sm.add_constant(x), y
    mod = sm.GLM(endog, exog, family=sm.families.Gamma(link=sm.families.links.log())) #log # identity
    res = mod.fit()

    sd=np.array([np.sqrt(res.cov_params()[0][0]),np.sqrt(res.cov_params()[1][1])])
    params2=res.params+2*sd
    params1=res.params-2*sd

    # print(res.summary())
    y_pred = res.predict(exog)
    y_pred_1=mod.predict(params1,exog)
    y_pred_2=mod.predict(params2,exog)

    idx = x.argsort()
    x_ord, y_pred_ord,y_pred_ord_1,y_pred_ord_2 = x[idx], y_pred[idx], y_pred_1[idx], y_pred_2[idx]



    # plt.plot(x_ord, y_pred_ord, color='m')
    # plt.scatter(x, y,  s=20, alpha=0.8)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    return x_ord, y_pred_ord,y_pred_ord_1,y_pred_ord_2


def plot_dissection_error(d13_df_mean,d13_df_sd,days,col='green'):
    ''' we plot CV vs log2(rel_abundance ) '''
    ### input parameters
    # d0_df_mean : day0 dataframe mean
    # d0_df_sd:   day0 datafreame SD
    # d13_df_mean day13
    # d13_df_sd day13

    ###
    ###  We would like to some generalized linear model statistics
    # x_fit_d0, y_fit_d0,y_fit_d0_1,y_fit_d0_2=applyGLM(d0_df_mean,d0_df_sd,day='d0',cut_sd=0.1,cut_mean=-14)
    # x_fit_d13, y_fit_d13,y_fit_d13_1,y_fit_d13_2=applyGLM(d13_df_mean,d13_df_sd,day='d13',cut_sd=0.1,cut_mean=-14)

    x_fit_d13, y_fit_d13,y_fit_d13_1,y_fit_d13_2=applyGLM(d13_df_mean,d13_df_sd,day=days[0],cut_sd=None,cut_mean=None)


    ###
    print('d13_dissection_mean=%.2f'%d13_df_mean.mean())

    fig = go.Figure()
    ### we are going to plot errors



    trace_d13 = go.Scatter(
        x = d13_df_mean[days[0]+'_mean'],
        y = d13_df_sd[days[0]+'_std'],
        mode = 'markers',
        marker=dict(size=5,color=col),
        name=days[0]+ '(# of markers=%d)'%d13_df_mean.shape[0],
        opacity=0.7,
        text=d13_df_mean['index'])

    trace_f13 = go.Scatter(
        x = x_fit_d13,
        y = y_fit_d13,
        line=dict(color=col, width=3),
        name=days[0]+'_fit',
        )

    trace_dist13 = go.Histogram( x=d13_df_mean[days[0]+'_mean'],
    name=days[0]+'_dist',
    marker_color='#bdbdbd',
    opacity=0.75)


    fig.add_trace(trace_d13)
    fig.add_trace(trace_f13)
    # fig.add_trace(trace_f13_1)
    # fig.add_trace(trace_f13_2)


    fig.update_layout(title='Dissection error analysis',xaxis_title="log2(relative_counts)",yaxis_title="Square of Coefficient of variations(CV*CV)",)
    # fig.show()
    disection_traces=[trace_d13,trace_f13,trace_dist13]
    return disection_traces



def plot_pcr_error(d0_df_mean,d0_df_sd,d13_df_mean,d13_df_sd,days):
    ''' we plot CV vs log2(rel_abundance ) '''
    ### input parameters
    # d0_df_mean : day0 dataframe mean
    # d0_df_sd:   day0 datafreame SD
    # d13_df_mean day13
    # d13_df_sd day13

    ###
    ###  We would like to some generalized linear model statistics
    # x_fit_d0, y_fit_d0,y_fit_d0_1,y_fit_d0_2=applyGLM(d0_df_mean,d0_df_sd,day='d0',cut_sd=0.1,cut_mean=-14)
    # x_fit_d13, y_fit_d13,y_fit_d13_1,y_fit_d13_2=applyGLM(d13_df_mean,d13_df_sd,day='d13',cut_sd=0.1,cut_mean=-14)

    x_fit_d0, y_fit_d0,y_fit_d0_1,y_fit_d0_2=applyGLM(d0_df_mean,d0_df_sd,day=days[0],cut_sd=None,cut_mean=None)
    x_fit_d13, y_fit_d13,y_fit_d13_1,y_fit_d13_2=applyGLM(d13_df_mean,d13_df_sd,day=days[1],cut_sd=None,cut_mean=None)


    ###

    print('d0_PCR_mean=%.2f'%d0_df_mean.mean())
    print('d13_PCR_mean=%.2f'%d13_df_mean.mean())

    fig = go.Figure()
    ### we are going to plot errors

    trace_d0 = go.Scatter(
        x = d0_df_mean[days[0]+'_mean'],
        y = d0_df_sd[days[0]+'_std'],
        mode = 'markers',
        marker=dict(
        size=5,
        color='red'), #set color equal to a variable
        # colorscale='Viridis', # one of plotly colorscales
        # showscale=True),
        name=days[0]+ '(# of markers=%d)'%d0_df_mean.shape[0],
        opacity=0.7,
        text=d0_df_mean['index'])


    trace_f0 = go.Scatter(
        x = x_fit_d0,
        y = y_fit_d0,
        line=dict(color='red', width=3),
        name=days[0]+'_fit',
        )

    trace_dist0 = go.Histogram( x=d0_df_mean[days[0]+'_mean'],
    name=days[0]+'_dist',
    marker_color='#bdbdbd',
    opacity=0.75)

    # trace_f0_1 = go.Scatter(
    #     x = x_fit_d0,
    #     y = y_fit_d0_1,
    #     line=dict(color='red', width=3,dash='dash'),
    #     name=days[0]+'_fit',
    #     )
    #
    # trace_f0_2 = go.Scatter(
    #     x = x_fit_d0,
    #     y = y_fit_d0_2,
    #     line=dict(color='red', width=3,dash='dash'),
    #     name=days[0]+'_fit',
    #     )



    trace_d13 = go.Scatter(
        x = d13_df_mean[days[1]+'_mean'],
        y = d13_df_sd[days[1]+'_std'],
        mode = 'markers',
        marker=dict(size=5,color='green'),
        name=days[1]+ '(# of markers=%d)'%d13_df_mean.shape[0],
        opacity=0.7,
        text=d13_df_mean['index'])

    trace_f13 = go.Scatter(
        x = x_fit_d13,
        y = y_fit_d13,
        line=dict(color='green', width=3),
        name=days[1]+'_fit',
        )

    trace_dist13 = go.Histogram( x=d13_df_mean[days[1]+'_mean'],
    name=days[1]+'_dist',
    marker_color='#bdbdbd',
    opacity=0.75)

    # trace_f13_1 = go.Scatter(
    #     x = x_fit_d13,
    #     y = y_fit_d13_1,
    #     line=dict(color='green', width=3,dash='dash'),
    #     name=days[1]+'_fit',
    #     )
    #
    # trace_f13_2 = go.Scatter(
    #     x = x_fit_d13,
    #     y = y_fit_d13_2,
    #     line=dict(color='green', width=3,dash='dash'),
    #     name=days[1]+'_fit',
    #     )

    fig.add_trace(trace_d0)

    fig.add_trace(trace_f0)
    # fig.add_trace(trace_f0_1)
    # fig.add_trace(trace_f0_2)
    fig.add_trace(trace_d13)
    fig.add_trace(trace_f13)
    # fig.add_trace(trace_f13_1)
    # fig.add_trace(trace_f13_2)


    fig.update_layout(title='PCR error analysis',xaxis_title="log2(relative_counts)",yaxis_title="Square of Coefficient of variations(CV*CV)",)

    # fig.show()
    pcr_tarces=[trace_d0,trace_f0,trace_d13,trace_f13,trace_dist0,trace_dist13]
    return pcr_tarces

def error_analysis(df,manfest_df,prev_to_new,db_df,plot_info=None):
    ''' We are going to do relative abundance analysis.
    With this analysis we will plot rlative abundance of each gene over time
    '''

    ########## input parameters ##############
    #### df: dataframe of count matrix
    ## manfest_df : manifest dataframe
    ##prev_to_new: previous pbnaka id to new pbanka id
    ##db_df : gene name and description
    # plot_info is not none then we would like to pass some variables for plot_info
    ##########  output ##############




    ###
    rel_df=df.copy()
    rel_df=rel_df.drop(columns=['Gene','Barcodes'])
    rel_df= rel_df + 1
    rel_df=rel_df.div(rel_df.sum(axis=1), axis=0)
    rel_df_log=np.log2(rel_df)


    ### convert old pbanka to new ids
    geneConv,old_to_new_ids,geneConv_new=getNewIdfromPrevID(rel_df.index,prev_to_new,db_df)
    rel_df=rel_df.rename(old_to_new_ids, axis='index')




    ###
    # we are going to find error byPCR
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    days=['d0','d13']
    listofvar=calculate_mean_sd_groupby_pcr(rel_df,manfest_df,grp_cols,day_pos,days) ## two days give 4 list of variables
    d0_df_mean,d0_df_sd,d13_df_mean,d13_df_sd=listofvar[0],listofvar[1],listofvar[2],listofvar[3]
    pcr_traces=plot_pcr_error(d0_df_mean,d0_df_sd,d13_df_mean,d13_df_sd,days)

    import pickle
    pickle.dump(d0_df_mean,open('rel_freq.pickle','wb'))


    #### diffrences between tube
    # we are going to find error byPCR
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    days=['NA']
    listofvar_tube=calculate_mean_sd_groupby_tube(rel_df,manfest_df,grp_cols,day_pos,days)
    tube_df_mean,tube_df_sd=listofvar_tube[0],listofvar_tube[1]
    tube_df_mean=tube_df_mean.rename(columns={"NA_mean": "cuvette_mean"})
    tube_df_sd=tube_df_sd.rename(columns={"NA_std": "cuvette_std"})

    days=['cuvette']
    tube_traces=plot_dissection_error(tube_df_mean,tube_df_sd,days,col='black')



    ## we would like to perform dissection errorx
    day_pos=grp_cols.index('d')
    days=['d13']
    listofvar_diss=calculate_mean_sd_groupby_dissection(rel_df,manfest_df,grp_cols,day_pos,days)
    d13_df_mean,d13_df_sd=listofvar_diss[0],listofvar_diss[1]

    disection_traces=plot_dissection_error(d13_df_mean,d13_df_sd,days,col='green')

    ## apply error analysis for the blood input

    ## we would like to perform blood (b1,b2,b3) error
    day_pos=grp_cols.index('d')
    days=['d0']
    grp_cols=['sex','d','mf','dr','b','t']
    listofvar=calculate_mean_sd_groupby_blood(rel_df,manfest_df,grp_cols,day_pos,days)

    d0_df_mean,d0_df_sd=listofvar[0],listofvar[1]
    blood_traces=plot_dissection_error(d0_df_mean,d0_df_sd,days,col='red')
    #### combine disection and PCR error
    # fig = go.Figure()
    ### we are going to plot errors

    ## we would like to perform mossifeed error
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    days=['d0']
    listofvar_feed=calculate_mean_sd_groupby_mossifeed(rel_df,manfest_df,grp_cols,day_pos,days)
    d13_df_mean,d13_df_sd=listofvar_feed[0],listofvar_feed[1]
    # df_m=d13_df_mean[d13_df_mean['d0_mean']<-11].copy()
    # df_s=d13_df_sd[d13_df_mean['d0_mean']<-11].copy()
    feed_traces=plot_dissection_error(d13_df_mean,d13_df_sd,days,col='red')
    day0_error_mean=d13_df_sd['d0_std'].mean()
    day0_error_var=d13_df_sd['d0_std'].std()
    cutoff_day0=day0_error_mean+3*day0_error_var

    # trace_bi=plot_dissection_error(df_m,df_s,days,col='red')
    # fig1 = make_subplots(rows=1, cols=1,subplot_titles=("Blood input error"))


    ### DO FOR D13
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    days=['d13']
    listofvar_feed=calculate_mean_sd_groupby_mossifeed_day13(rel_df,manfest_df,grp_cols,day_pos,days)
    out_d13_df_mean,out_d13_df_sd=listofvar_feed[0],listofvar_feed[1]
    # df_m_d13=out_d13_df_mean[out_d13_df_mean['d13_mean']<-11].copy()
    # df_s_d13=out_d13_df_sd[out_d13_df_mean['d13_mean']<-11].copy()
    day13_error_mean=out_d13_df_sd['d13_std'].mean()
    day13_error_var=out_d13_df_sd['d13_std'].std()
    cutoff_day13=day13_error_mean+3*day13_error_var

    ## now we are going to collect untrusted mutanats
    filteredIdx=(d13_df_mean['d0_mean']<-12.2) | ((d13_df_mean['d0_mean']<-10.5) & (d13_df_sd['d0_std']> cutoff_day0))
    day0_filtered_mean=d13_df_mean[filteredIdx].copy()
    day0_filtered_var=d13_df_sd[filteredIdx].copy()

    filteredIdx=(out_d13_df_mean['d13_mean']<-12.2) | ((out_d13_df_mean['d13_mean']<-10.5) & (out_d13_df_sd['d13_std']> cutoff_day13))
    day13_filtered_mean=out_d13_df_mean[filteredIdx].copy()
    day13_filtered_var=out_d13_df_sd[filteredIdx].copy()

    cmb_df=pd.concat([day0_filtered_mean,day0_filtered_var],axis=1)
    cmb_df.to_csv('filtered genes and conditions_d0.txt',sep='\t',index=None)

    cmb_df=pd.concat([day13_filtered_mean,day13_filtered_var],axis=1)
    cmb_df.to_csv('filtered genes and conditions_d13.txt',sep='\t',index=None)


    # fig1.append_trace(trace_bi[0],row=1, col=1)
    # fig1.update_yaxes(title_text="relative error (CV*CV)",row=1, col=1)
    # fig1.update_xaxes(title_text="log2 (relative abundance or count)", row=1, col=1)
    # fig1.show()

    # fig.append_trace(trace_bi[1],row=6, col=1)
    # fig.append_trace(trace_bi[2],row=6, col=2)


    fig = make_subplots(rows=6, cols=2,subplot_titles=("PCR error", "Distribution", "PCR error","Distribution", "Blood input error", "Distribution",
    "Dissection error","Distribution","Cuvette error", "Distribution","Mosquito feed error", "Distribution"))

    fig.append_trace(pcr_traces[0],row=1, col=1)

    fig.append_trace(pcr_traces[1],row=1, col=1)
    fig.append_trace(pcr_traces[4],row=1, col=2)
    fig.append_trace(pcr_traces[2],row=2, col=1)
    fig.append_trace(pcr_traces[3],row=2, col=1)
    fig.append_trace(pcr_traces[5],row=2, col=2)


    fig.append_trace(blood_traces[0],row=3, col=1)
    fig.append_trace(blood_traces[1],row=3, col=1)
    fig.append_trace(blood_traces[2],row=3, col=2)
    # fig.append_trace(blood_traces[2],row=2, col=1)
    # fig.append_trace(blood_traces[3],row=2, col=1)

    fig.append_trace(disection_traces[0],row=4, col=1)
    fig.append_trace(disection_traces[1],row=4, col=1)
    fig.append_trace(disection_traces[2],row=4, col=2)

    fig.append_trace(tube_traces[0],row=5, col=1)
    fig.append_trace(tube_traces[1],row=5, col=1)
    fig.append_trace(tube_traces[2],row=5, col=2)

    fig.append_trace(feed_traces[0],row=6, col=1)
    fig.append_trace(feed_traces[1],row=6, col=1)
    fig.append_trace(feed_traces[2],row=6, col=2)

    # Update xaxis properties
    fig.update_yaxes(title_text="relative error (CV*CV)",row=1, col=1,range=[0,1.6])
    fig.update_yaxes(title_text="relative error (CV*CV)",row=2, col=1,range=[0,1.6])
    fig.update_yaxes(title_text="relative error (CV*CV)"    ,row=3, col=1,range=[0,1.6])
    fig.update_yaxes(title_text="relative error (CV*CV)"    ,row=4, col=1,range=[0,1.6])
    fig.update_yaxes(title_text="relative error (CV*CV)"    ,row=5, col=1,range=[0,1.6])
    fig.update_yaxes(title_text="relative error (CV*CV)"    ,row=6, col=1,range=[0,1.6])

    fig.update_yaxes(title_text="Frequency"    ,row=1, col=2)
    fig.update_yaxes(title_text="Frequency"     ,row=2, col=2)
    fig.update_yaxes(title_text="Frequency"    ,row=3, col=2)
    fig.update_yaxes(title_text="Frequency"     ,row=4, col=2)
    fig.update_yaxes(title_text="Frequency"     ,row=5, col=2)
    fig.update_yaxes(title_text="Frequency"     ,row=6, col=2)

    # Update yaxis properties
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=1, col=1,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=2, col=1,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=3, col=1,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=4, col=1,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=5, col=1,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=6, col=1,range=[-20,0])

    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=1, col=2,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=2, col=2,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=3, col=2,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=4, col=2,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=5, col=2,range=[-20,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=6, col=2,range=[-20,0])

    fig.update_layout(height=2700, width=1200, title_text="Diffrent types of error analysis")
    fig.show()



def plot_propgated_relative_abunndance(mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2,mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2,geneConv,plot_info):
    ''' We will plot genes sex-specific wise '''

    #### input ####
    # sex_list: sex_list[0]=mean and sex_list[1]=SD
    # geneConv: when we want to give name of gene
    # out_pdf: will generated pdf file
    ########
    out_pdf=plot_info['rel_file']
    mf=plot_info['mf']
    day=plot_info['d']
    sex=plot_info['sex']

    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    n=2
    genes=mean_df_d0_mf1.index;

    labels=[]
    for gene in genes:
        if gene in geneConv.keys():
            labels.append(geneConv[gene])
        else:
            labels.append(gene)

    ### get number of subplots

    num_subplots=2*len(genes) ### number of subplots

    l=len(genes)
    loops=int(l/(n*n))
    rem=l % (n*n)
    k=0

    per_page=num_subplots/16

    for loop in range(loops):
        fig = plt.figure(figsize=(15,15))
        for i in range(1,(n*n)+1):

            plt.subplot(n, n, i)
            x=[1,2]


            ### mf1 female
            y=[mean_df_d0_mf1.loc[genes[k],sex[0]+'_'+day[0]+'_NA_NA'],mean_df_d13_mf1.loc[genes[k],sex[0]+'_'+day[1]+'_NA_NA']]
            yerr=[var_df_d0_mf1.loc[genes[k],sex[0]+'_'+day[0]+'_NA_NA'],var_df_d13_mf1.loc[genes[k],sex[0]+'_'+day[1]+'_NA_NA']]
            plt.errorbar(x,y, yerr=yerr, fmt='r--')

            ### mf2 female
            y=[mean_df_d0_mf2.loc[genes[k],sex[0]+'_'+day[0]+'_NA_NA'],mean_df_d13_mf2.loc[genes[k],sex[0]+'_'+day[1]+'_NA_NA']]
            yerr=[var_df_d0_mf2.loc[genes[k],sex[0]+'_'+day[0]+'_NA_NA'],var_df_d13_mf2.loc[genes[k],sex[0]+'_'+day[1]+'_NA_NA']]
            plt.errorbar(x,y, yerr=yerr, fmt='r-')

            ### mf1 male
            y=[mean_df_d0_mf1.loc[genes[k],sex[1]+'_'+day[0]+'_NA_NA'],mean_df_d13_mf1.loc[genes[k],sex[1]+'_'+day[1]+'_NA_NA']]
            yerr=[var_df_d0_mf1.loc[genes[k],sex[1]+'_'+day[0]+'_NA_NA'],var_df_d13_mf1.loc[genes[k],sex[1]+'_'+day[1]+'_NA_NA']]
            plt.errorbar(x,y, yerr=yerr, fmt='b--')

            ### mf2 male
            y=[mean_df_d0_mf2.loc[genes[k],sex[1]+'_'+day[0]+'_NA_NA'],mean_df_d13_mf2.loc[genes[k],sex[1]+'_'+day[1]+'_NA_NA']]
            yerr=[var_df_d0_mf2.loc[genes[k],sex[1]+'_'+day[0]+'_NA_NA'],var_df_d13_mf2.loc[genes[k],sex[1]+'_'+day[1]+'_NA_NA']]
            plt.errorbar(x,y, yerr=yerr, fmt='b-')


            plt.ylabel('log2 relative fitness')
            plt.title(labels[k])
            plt.legend(('mf1_GCKO2', 'mf2_GCKO2','mf1_145480', 'mf2_145480'))
            plt.xticks([1, 1.5, 2],['day0', '', 'day13'],fontsize=15)

            plt.ylim(-18, 1)
            plt.grid(False)
            k=k+1
        pdf.savefig(fig)

    ## for the remaing one

    fig = plt.figure(figsize=(15,15))
    for i in range(1,rem+1):



        plt.subplot(n, n, i)
        x=[1,2]


        ### mf1 female
        y=[mean_df_d0_mf1.loc[genes[k],sex[0]+'_'+day[0]+'_NA_NA'],mean_df_d13_mf1.loc[genes[k],sex[0]+'_'+day[1]+'_NA_NA']]
        yerr=[var_df_d0_mf1.loc[genes[k],sex[0]+'_'+day[0]+'_NA_NA'],var_df_d13_mf1.loc[genes[k],sex[0]+'_'+day[1]+'_NA_NA']]
        plt.errorbar(x,y, yerr=yerr, fmt='r--')

        ### mf2 female
        y=[mean_df_d0_mf2.loc[genes[k],sex[0]+'_'+day[0]+'_NA_NA'],mean_df_d13_mf2.loc[genes[k],sex[0]+'_'+day[1]+'_NA_NA']]
        yerr=[var_df_d0_mf2.loc[genes[k],sex[0]+'_'+day[0]+'_NA_NA'],var_df_d13_mf2.loc[genes[k],sex[0]+'_'+day[1]+'_NA_NA']]
        plt.errorbar(x,y, yerr=yerr, fmt='r-')

        ### mf1 male
        y=[mean_df_d0_mf1.loc[genes[k],sex[1]+'_'+day[0]+'_NA_NA'],mean_df_d13_mf1.loc[genes[k],sex[1]+'_'+day[1]+'_NA_NA']]
        yerr=[var_df_d0_mf1.loc[genes[k],sex[1]+'_'+day[0]+'_NA_NA'],var_df_d13_mf1.loc[genes[k],sex[1]+'_'+day[1]+'_NA_NA']]
        plt.errorbar(x,y, yerr=yerr, fmt='b--')

        ### mf2 male
        y=[mean_df_d0_mf2.loc[genes[k],sex[1]+'_'+day[0]+'_NA_NA'],mean_df_d13_mf2.loc[genes[k],sex[1]+'_'+day[1]+'_NA_NA']]
        yerr=[var_df_d0_mf2.loc[genes[k],sex[1]+'_'+day[0]+'_NA_NA'],var_df_d13_mf2.loc[genes[k],sex[1]+'_'+day[1]+'_NA_NA']]
        plt.errorbar(x,y, yerr=yerr, fmt='b-')



        plt.ylabel('log2 relative fitness')
        plt.title(labels[k])
        plt.legend(('mf1_GCKO2', 'mf2_GCKO2','mf1_g145480', 'mf2_g145480'))
        plt.xticks([1, 1.5, 2],['day0', '', 'day13'],fontsize=15)

        plt.ylim(-18, 1)
        plt.grid(False)
        k=k+1
        pdf.savefig(fig)
    pdf.close()


def getNewIdfromPrevID(pgenes,prev_to_new,db_df):

    ''' this is used for gene id conversion '''

    not_found=[]
    geneConv={}  ### old Id to description
    geneConv_new={} ### new Id to description
    old_to_new_ids={} ### old_id  to new_id


    for item in pgenes:
        if item in prev_to_new.keys():
            tmp= db_df[db_df['Gene ID']==prev_to_new[item]].copy()
            if tmp.empty:
                not_found.append(item)
                geneConv[item]=item
                old_to_new_ids[item]=item
                geneConv_new[item]=item
            else:

                if tmp['Gene Name or Symbol'].to_list()[0]=='NA':
                    geneConv[item]=tmp['Product Description'].to_list()[0]+'|'+prev_to_new[item]
                    old_to_new_ids[item]=prev_to_new[item]
                    geneConv_new[prev_to_new[item]]=tmp['Product Description'].to_list()[0]+'|'+prev_to_new[item]
                else:
                    geneConv[item]=tmp['Product Description'].to_list()[0]+'|'+tmp['Gene Name or Symbol'].to_list()[0]+'|'+prev_to_new[item]
                    old_to_new_ids[item]=prev_to_new[item]
                    geneConv_new[prev_to_new[item]]=tmp['Product Description'].to_list()[0]+'|'+tmp['Gene Name or Symbol'].to_list()[0]+'|'+prev_to_new[item]
    print('not found genes new ids for old gene ids ', not_found)
    return geneConv,old_to_new_ids,geneConv_new





def error_analysis_mosquito_feed(df,manfest_df,prev_to_new,db_df,plot_info=None):
    ''' We are going to do relative abundance analysis.
    With this analysis we will plot rlative abundance of each gene over time
    '''

    ########## input parameters ##############
    #### df: dataframe of count matrix
    ## manfest_df : manifest dataframe
    ##prev_to_new: previous pbnaka id to new pbanka id
    ##db_df : gene name and description
    # plot_info is not none then we would like to pass some variables for plot_info
    ##########  output ##############




    ###
    rel_df=df.copy()
    rel_df=rel_df.drop(columns=['Gene','Barcodes'])
    rel_df= rel_df + 1
    rel_df=rel_df.div(rel_df.sum(axis=1), axis=0)
    rel_df_log=np.log2(rel_df)


    ### convert old pbanka to new ids
    geneConv,old_to_new_ids,geneConv_new=getNewIdfromPrevID(rel_df.index,prev_to_new,db_df)
    rel_df=rel_df.rename(old_to_new_ids, axis='index')




    ###

    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    days=['d0']
    listofvar_feed=calculate_mean_sd_groupby_mossifeed(rel_df,manfest_df,grp_cols,day_pos,days)
    d13_df_mean,d13_df_sd=listofvar_feed[0],listofvar_feed[1]
    # df_m=d13_df_mean[d13_df_mean['d0_mean']<-11].copy()
    # df_s=d13_df_sd[d13_df_mean['d0_mean']<-11].copy()
    feed_traces=plot_dissection_error(d13_df_mean,d13_df_sd,days,col='red')
    day0_error_mean=d13_df_sd['d0_std'].mean()
    day0_error_var=d13_df_sd['d0_std'].std()
    cutoff_day0=day0_error_mean+3*day0_error_var

    # trace_bi=plot_dissection_error(df_m,df_s,days,col='red')
    # fig1 = make_subplots(rows=1, cols=1,subplot_titles=("Blood input error"))

    # import pdb;pdb.set_trace()
    ### DO FOR D13
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    days=['d13']
    listofvar_feed=calculate_mean_sd_groupby_mossifeed_day13(rel_df,manfest_df,grp_cols,day_pos,days)
    out_d13_df_mean,out_d13_df_sd=listofvar_feed[0],listofvar_feed[1]
    # df_m_d13=out_d13_df_mean[out_d13_df_mean['d13_mean']<-11].copy()
    # df_s_d13=out_d13_df_sd[out_d13_df_mean['d13_mean']<-11].copy()
    day13_error_mean=out_d13_df_sd['d13_std'].mean()
    day13_error_var=out_d13_df_sd['d13_std'].std()
    cutoff_day13=day13_error_mean+3*day13_error_var

    ## now we are going to collect untrusted mutanats
    filteredIdx=(d13_df_mean['d0_mean']<-12.2) | ((d13_df_mean['d0_mean']<-10.5) & (d13_df_sd['d0_std']> cutoff_day0))
    day0_filtered_mean=d13_df_mean[filteredIdx].copy()
    day0_filtered_var=d13_df_sd[filteredIdx].copy()

    filteredIdx=(out_d13_df_mean['d13_mean']<-12.2) | ((out_d13_df_mean['d13_mean']<-10.5) & (out_d13_df_sd['d13_std']> cutoff_day13))
    day13_filtered_mean=out_d13_df_mean[filteredIdx].copy()
    day13_filtered_var=out_d13_df_sd[filteredIdx].copy()

    cmb_df=pd.concat([day0_filtered_mean,day0_filtered_var],axis=1)
    cmb_df.to_csv('filtered genes and conditions_d0.txt',sep='\t',index=None)

    cmb_df=pd.concat([day13_filtered_mean,day13_filtered_var],axis=1    )
    cmb_df.to_csv('filtered genes and conditions_d13.txt',sep='\t',index=None)




def apply_filter_testInput(pheno_call_df,mean_df_d0_mf1,mean_df_d0_mf2,rel_cut=-12,sex=['GCKO2','g145480']):
    ''' test diffrent level of errors and comapre with combined errors'''
    for s in sex:
        aa=mean_df_d0_mf1.columns.str.contains(s)
        if aa.any():
            mf1_gene=mean_df_d0_mf1[mean_df_d0_mf1.iloc[:,np.where(aa)[0][0]]<rel_cut].index
        aa=mean_df_d0_mf1.columns.str.contains(s)
        if aa.any():
            mf2_gene=mean_df_d0_mf2[mean_df_d0_mf2.iloc[:,np.where(aa)[0][0]]<rel_cut].index
        ## get common genes

        comm=set(mf1_gene) & set(mf2_gene)
        only_mf1=set(mf1_gene) - set(mf2_gene)
        only_mf2=set(mf2_gene) - set(mf1_gene)


        if len(comm)>0:
            # pheno_call_df.loc[comm,s+'_pheno']='No Power'
            pheno_call_df.loc[comm,s+'_pheno']='No Data'
            pheno_call_df.loc[comm,s+'_relative_filter']='applied'
            pheno_call_df.loc[comm,s+'_feed']='no data'
        if len(only_mf1)>0:
            pheno_call_df.loc[only_mf1,s+'_feed']='Mf2'

        if len(only_mf2)>0:
            pheno_call_df.loc[only_mf2,s+'_feed']='Mf1'

    return pheno_call_df





def plot_gDNA_error(m_df, v_df):
    sex=['GCKO2','g145480']
    feed=['mf1','mf2']
    gDNA=['p','q']
    cols=m_df.columns
    titles=m_df.columns.str.replace('NA_NA','')
    count=0
    fig = make_subplots(rows=4, cols=3,subplot_titles=tuple(titles))

    for k in range(4):
        for i in range(2):
            x=cols[count]
            y=cols[count]
            count=count+1

            trace = go.Scatter(
            x = m_df[x],
            y = v_df[x],
            mode = 'markers',
            marker=dict(size=5,color='red'),
            name='(# of markers=%d)'%m_df.shape[0],
            opacity=0.7,
            text=m_df.index)
            fig.append_trace(trace,row=k+1, col=i+1)

            # Update xaxis properties
            # fig.update_yaxes(title_text="relative error",row=1, col=1,range=[0,0.8])
            fig.update_yaxes(title_text="relative error",row=k+1, col=i+1,range=[0,0.8])
            fig.update_xaxes(title_text="relative abundance",row=k+1, col=i+1)


        trace = go.Scatter(
        x = m_df[cols[count-1]],
        y = m_df[cols[count-2]],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%m_df.shape[0],
        opacity=0.7,
        text=m_df.index)
        fig.append_trace(trace,row=k+1, col=3)

        # Update xaxis properties
        # fig.update_yaxes(title_text="relative error",row=1, col=1,range=[0,0.8])
        fig.update_yaxes(title_text="relative abundance (q)",row=k+1, col=3,range=[-15,0])
        fig.update_xaxes(title_text="relative abundance (p)",row=k+1, col=3,range=[-15,0])

            #
    fig.update_layout(height=2000, width=1500, title_text="gDNA extraction error analysis")
    fig.show()



def getPvalZscore(cmb_fitness,upcut=1,lowcut=0.1,pval=0.01,pval1=0.01):
    # cmb_fitness['GCKO'][0] is RGR final
    # cmb_fitness['GCKO'][1] is varaiace final
    ## upcut: for Non essential
    ## lowcut: for essential
    ## pval : for non essential
    ## pval1: for essential

    for k in cmb_fitness.keys():
        key1=k
        break

    viz_df=pd.DataFrame(index=cmb_fitness[key1][0].index)

    upcut=np.log2(upcut)
    lowcut=np.log2(lowcut)

    # pheno_pval={}

    for key,item in cmb_fitness.items():

    # we will calculate p-vlaue and z score

        m=item[0]
        s=item[1]
        z=(upcut-m)/s
        df = pd.DataFrame(m, columns=[key])
        df[(key,'SD')]=s
        df['z_score']=z
        # df['pvalue']=1-sp.ndtr(df['z_score'])
        # df['pvalue_2tail'] = 2*(1 - sp.ndtr(df['z_score']))
        df['pvalue']=  (1 - st.norm.cdf(z))

       # we will calculate p value form 0.1
        z = (lowcut - m) / s
        df['z_score_2'] = z
        df['pvalue_2'] = (1 - st.norm.cdf(z))

        # select fast growing, NE, E, middle
        # if pvalue<0.05 and relative fitness >0
        df_fast=df[(df['pvalue'] < pval) & (df[key] > upcut)]
        df_ambiguous1=df[(df['pvalue'] < pval) & (df[key] < upcut)]
        NE_df=df[(df['pvalue'] > pval)]

        df_subset1= df[(df['pvalue_2'] >pval1)]
        df_ambiguous2 = df[(df['pvalue_2'] < pval1) & (df[key] > lowcut)]
        E_df=df[(df['pvalue_2'] < pval1)]
        # non essential genes
        NE=(NE_df.index)
        # essential genes
        E=E_df.index
        # dropIdx=NE.union(E)
        # # other genes
        # df_other=df.drop(index=dropIdx)
        #
        # other=df_other.index

        viz_df[key+'_pheno'] = 'NA'
        viz_df.loc[E,key+'_pheno'] = 'E'
        viz_df.loc[NE,key+'_pheno'] = 'NE'
        viz_df[key+'_RGR'] = m
        viz_df[key+'_sd'] = s
        viz_df[key+'_var'] = item[2]

        # pheno_pval['E']=E
        # pheno_pval['NE'] = NE
        # pheno_pval['other'] = other
        #import pdb;pdb.set_trace()  # we are checking the pvalues for normal genes .

    return viz_df




def testforRepeat(pheno_call_df,mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2,mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2,cuvette_mean_df,cuvette_var_df,rel_cut=-10.5,plot_info=None):
    ''' test diffrent level of errors and comapre with combined errors'''

    cutoff_zero=-13.5 ### when relative abundance is too small


    input=[mean_df_d0_mf1,mean_df_d0_mf2]
    input_var=[var_df_d0_mf1,var_df_d0_mf2]
    output=[mean_df_d13_mf1,mean_df_d13_mf2]
    output_var=[var_df_d13_mf1,var_df_d13_mf2]
    feeds=[]
    # backgrounds=['GCKO2','g145480']

    result=pd.DataFrame(index=mean_df_d0_mf1.index)
    result2=pd.DataFrame(index=mean_df_d0_mf1.index)

    for i,f in enumerate(['mf1','mf2']):
        for col in input[i].columns:
            result[col+ '_'+f+'_mean']=np.nan
            result[col+ '_'+f+'_var']=np.nan
            result2[col+ '_'+f+'_mean']=np.nan
            result2[col+ '_'+f+'_var']=np.nan
            # result2=result.copy()

            result2.loc[:,col+'_'+f+'_mean']=input[i].loc[:,col].copy()
            result2.loc[:,col+'_'+f+'_var']=input_var[i].loc[:,col].copy()

            m=input_var[i].loc[:,col].mean()
            s=input_var[i].loc[:,col].std()


            s1=set(input[i].loc[input[i].loc[:,col]<rel_cut,:].index)
            s2=set(input_var[i].loc[input_var[i].loc[:,col]> m+2*s,:].index)
            s3=set(input[i].loc[input[i].loc[:,col]<cutoff_zero,:].index)
            genes=(s1&s2)|s3
            result.loc[genes,col+ '_'+f+'_mean']=input[i].loc[genes,col].copy()
            result.loc[genes,col+ '_'+f+'_var']=input_var[i].loc[genes,col].copy()

    for i,f in enumerate(['mf1','mf2']):
        for col in output[i].columns:
            result[col+ '_'+f+'_mean']=np.nan
            result[col+ '_'+f+'_var']=np.nan
            result2[col+ '_'+f+'_mean']=np.nan
            result2[col+ '_'+f+'_var']=np.nan

            result2.loc[:,col+'_'+f+'_mean']=output[i].loc[:,col].copy()
            result2.loc[:,col+'_'+f+'_var']=output_var[i].loc[:,col].copy()

            m=output_var[i].loc[:,col].mean()
            s=output_var[i].loc[:,col].std()


            s1=set(output[i].loc[output[i].loc[:,col]<rel_cut,:].index)
            s2=set(output_var[i].loc[output_var[i].loc[:,col]> m+2*s,:].index)
            s3=set(output[i].loc[output[i].loc[:,col]<cutoff_zero,:].index)
            genes=(s1&s2)|s3
            result.loc[genes,col+ '_'+f+'_mean']=output[i].loc[genes,col].copy()
            result.loc[genes,col+ '_'+f+'_var']=output_var[i].loc[genes,col].copy()




    # # for combined fitness
    rgr_cols=['GCKO2_RGR', 'g145480_RGR']
    var_cols=['GCKO2_var', 'g145480_var']
    for i,col in enumerate(rgr_cols):
        result['final_'+col]=np.nan
        result['final_'+var_cols[i]]=np.nan

        result2['final_'+col]=pheno_call_df.loc[:,col].copy()
        result2['final_'+var_cols[i]]=pheno_call_df.loc[:,var_cols[i]].copy()
        m=pheno_call_df.loc[:,var_cols[i]].mean()
        s=pheno_call_df.loc[:,var_cols[i]].std()
        # s1=set(pheno_call_df.loc[pheno_call_df.loc[:,col]<rel_cut,:].index)
        s2=set(pheno_call_df.loc[pheno_call_df.loc[:,var_cols[i]]> m+2*s,:].index)
        # s3=set(pheno_call_df.loc[pheno_call_df.loc[:,col]<cutoff_zero,:].index)
        genes=s2
        result.loc[genes,'final_'+col]=pheno_call_df.loc[genes,col].copy()
        result.loc[genes,'final_'+var_cols[i]]=pheno_call_df.loc[genes,var_cols[i]].copy()
    # result2.loc[:,col+'_'+f+'_mean']=output[i][result2.index,col].copy()
    # result2.loc[:,col+'_'+f+'_var']=output_var[i][result2.index,col].copy()
    # # for col in pheno_call_df

    ## for cuvette analysis

    for col in cuvette_mean_df.columns:
        result[col+ '_cuvette_mean']=np.nan
        result[col+ '_cuvette_var']=np.nan

        result2[col+ '_cuvette_mean']=cuvette_mean_df.loc[:,col].copy()
        result2[col+ '_cuvette_var']=cuvette_var_df.loc[:,col].copy()



        m=cuvette_var_df.loc[:,col].mean()
        s=cuvette_var_df.loc[:,col].std()


        s1=set(cuvette_mean_df.loc[cuvette_mean_df.loc[:,col]<rel_cut,:].index)
        s2=set(cuvette_var_df.loc[cuvette_var_df.loc[:,col]> m+2*s,:].index)
        s3=set(cuvette_mean_df.loc[cuvette_mean_df.loc[:,col]<cutoff_zero,:].index)
        genes=(s1&s2)|s3
        result.loc[genes,col+ '_'+f+'_mean']=cuvette_mean_df.loc[genes,col].copy()
        result.loc[genes,col+ '_'+f+'_var']=cuvette_var_df.loc[genes,col].copy()

    result=result.dropna(how='all')

    if 'file' in plot_info.keys():
        with pd.ExcelWriter(plot_info['file']) as writer:
            result.to_excel(writer, sheet_name='filter')
            result2.to_excel(writer, sheet_name='all')


    print('repeat is done')


def relative_growth_rate_analysis(df,manfest_df,prev_to_new,db_df,plot_info=None):
    ''' We are going to do relative growth rate analysis'''

    ## first we need to propagate error from PCR to mosquito feed

    rel_df=df.copy()
    rel_df=rel_df.drop(columns=['Gene','Barcodes'])
    rel_df= rel_df + 1
    rel_df=rel_df.div(rel_df.sum(axis=0), axis=1)
    rel_df_log=np.log2(rel_df)


    ### convert old pbanka to new ids
    geneConv,old_to_new_ids,geneConv_new=getNewIdfromPrevID(rel_df.index,prev_to_new,db_df)
    rel_df=rel_df.rename(old_to_new_ids, axis='index')

    # newIds for control genes
    control_genes= plot_info['control_genes']
    ctrl_genes=[]
    for c in control_genes:
        if c in prev_to_new.keys():
            ctrl_genes.append(prev_to_new[c])
        else:
            ctrl_genes.append(c)



    ## now we are going to propagate error
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')

    mean_df_d0,var_df_d0=propagate_error_day0(rel_df,manfest_df,grp_cols,day_pos)
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    mean_df_d13,var_df_d13=propagate_error_day13(rel_df,manfest_df,grp_cols,day_pos)

    #### now we need to tarnsform at log2 scale
    mean_df_d0_log=np.log2(mean_df_d0)
    var_df_d0_log=(var_df_d0)/((mean_df_d0*mean_df_d0)*((np.log(10)**2)*np.log10(2)))

    mean_df_d13_log=np.log2(mean_df_d13)
    var_df_d13_log=(var_df_d13)/((mean_df_d13*mean_df_d13)*((np.log(10)**2)*np.log10(2)))

    ### we are going to compute each mosquito feed

    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2=propagate_error_day0_each_mossifeed(rel_df,manfest_df,grp_cols,day_pos)
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2=propagate_error_day13_each_mossifeed(rel_df,manfest_df,grp_cols,day_pos)

    saveres=[[mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2],[mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2]]

    pickle.dump(saveres,open(plot_info['pool']+'.p','wb'))
    ##  we are going to compute rleative growth rate  ###

    ### plot propagated relative abundance.
    propagated_relative_baundance_plot(mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2,mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2,plot_info)

    ## propgate error for transfection cuvette
    grp_cols=['sex','d','mf','dr','b','t']
    day_pos=grp_cols.index('d')
    days=['NA']
    cuvette_mean_df,cuvette_var_df=propagate_error_cuvette(rel_df,manfest_df,grp_cols,day_pos,days)
    ##

    mf1_RGR,mf1_var=calculate_RGR(mean_df_d0_mf1.copy(),var_df_d0_mf1.copy(),mean_df_d13_mf1.copy(),var_df_d13_mf1.copy(),ctrl_genes)
    mf2_RGR,mf2_var=calculate_RGR(mean_df_d0_mf2.copy(),var_df_d0_mf2.copy(),mean_df_d13_mf2.copy(),var_df_d13_mf2.copy(),ctrl_genes)
    ### now combined fitness for mf1 and mf2
    # take mf1 and mf2 in one dtaframe


    cmb_fitness={}
    backgrounds=['GCKO2','g145480']

    for b in backgrounds:
        rgr_temp=pd.DataFrame(index=mf1_RGR.index,columns=['mf1','mf2'])
        var_temp=pd.DataFrame(index=mf1_RGR.index,columns=['mf1','mf2'])
        col_mf1 = getColumnsFormDF(mf1_RGR, [b])
        col_mf2 = getColumnsFormDF(mf2_RGR, [b])

        rgr_temp.loc[:,'mf1']=mf1_RGR.loc[:,col_mf1[0].to_list()[0]].copy()
        rgr_temp.loc[:,'mf2']=mf2_RGR.loc[:,col_mf2[0].to_list()[0]].copy()
        var_temp.loc[:,'mf1']=mf1_var.loc[:,col_mf1[0].to_list()[0]].copy()
        var_temp.loc[:,'mf2']=mf2_var.loc[:,col_mf2[0].to_list()[0]].copy()


        cmb_fitness[b]=gaussianMeanAndVariance(rgr_temp,var_temp)

    ## calculate combined fitness
    pheno_call_df=applyPhenocall_CI(cmb_fitness,lower_cut=np.log2(0.45),upper_cut=np.log2(2.05))
    # pheno_call_df=getPvalZscore(cmb_fitness,upcut=1,lowcut=0.4,pval=0.05,pval1=0.05)

    ### aplly filter of relative input
    pheno_call_df=apply_filter_testInput(pheno_call_df,mean_df_d0_mf1,mean_df_d0_mf2,rel_cut=-12)

    ### once again apply filter on feeds
    pheno_call_df2=apply_filter_on_feeds(pheno_call_df,mf1_RGR,mf1_var,mf2_RGR,mf2_var,mean_df_d0_mf1,mean_df_d0_mf2)

    # test variance of comined and variance with each step using cutoff
    # rel_cut=-12
    # testforRepeat(pheno_call_df,mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2,mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2,cuvette_mean_df,cuvette_var_df,rel_cut,plot_info)

    ## for pool2 and pool4

    if (plot_info['pool']=='pool2') or (plot_info['pool']=='pool4'):

        grp_cols=['sex','d','mf','dr','e','t']
        day_pos=grp_cols.index('d')
        gDNA_mean_df, gDNA_mean_df_var=propagate_error_gDNA_extraction_method(rel_df,manfest_df,grp_cols,day_pos)
        plot_gDNA_error(gDNA_mean_df, gDNA_mean_df_var)
    ##

    plot_each_step_mean_var(mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2,mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2)

    ## compute relative ratio betwen day13 and day0
    trace_d0_female = go.Scatter(
        x = mean_df_d0_log['GCKO2_d0_NA_NA'],
        y = var_df_d0_log['GCKO2_d0_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d0_log.shape[0],
        opacity=0.7,
        text=mean_df_d0_log.index)

    trace_d0_male = go.Scatter(
        x = mean_df_d0_log['g145480_d0_NA_NA'],
        y = var_df_d0_log['g145480_d0_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d0_log.shape[0],
        opacity=0.7,
        text=mean_df_d0_log.index)

    trace_d13_female = go.Scatter(
        x = mean_df_d13_log['GCKO2_d13_NA_NA'],
        y = var_df_d13_log['GCKO2_d13_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d13_log.shape[0],
        opacity=0.7,
        text=mean_df_d13_log.index)

    trace_d13_male = go.Scatter(
        x = mean_df_d13_log['g145480_d13_NA_NA'],
        y = var_df_d13_log['g145480_d13_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d13_log.shape[0],
        opacity=0.7,
        text=mean_df_d13_log.index)

    fig = make_subplots(rows=2, cols=2,subplot_titles=("Input error GCKO2(day0)", "Input error 145480(day0)","Output error GCKO2(day13)","Output error 145480(day13)"))

    fig.append_trace(trace_d0_female,row=1, col=1)
    fig.append_trace(trace_d0_male,row=1, col=2)
    fig.append_trace(trace_d13_female,row=2, col=1)

    fig.append_trace(trace_d13_male,row=2, col=2)

    # fig.show()

    return pheno_call_df2,[mean_df_d0_mf1,mean_df_d0_mf2]

def apply_filter_on_feeds(pheno_call_df,mf1_RGR,mf1_var,mf2_RGR,mf2_var,mean_df_d0_mf1,mean_df_d0_mf2, backgrounds=['GCKO2','g145480']):
    ''' apply_filter_on_feeds '''


    cmb_fit_final={}


    for b in backgrounds:
        cmb_fitness={}
        col_mf1 = getColumnsFormDF(mf1_RGR, [b])
        col_mf2 = getColumnsFormDF(mf2_RGR, [b])



        ### select both feeds

        pheno_both=pheno_call_df[(pheno_call_df[b+'_feed']=='Mf1 and Mf2')| (pheno_call_df[b+'_feed']=='no data')]
        pheno_mf1=pheno_call_df[pheno_call_df[b+'_feed']=='Mf1']
        pheno_mf2=pheno_call_df[pheno_call_df[b+'_feed']=='Mf2']

        if len(pheno_both.index)>0:
            rgr_temp=pd.DataFrame(index=mf1_RGR.index,columns=['mf1','mf2'])
            var_temp=pd.DataFrame(index=mf1_RGR.index,columns=['mf1','mf2'])
            rgr_temp.loc[:,'mf1']=mf1_RGR.loc[:,col_mf1[0].to_list()[0]].copy()
            rgr_temp.loc[:,'mf2']=mf2_RGR.loc[:,col_mf2[0].to_list()[0]].copy()
            var_temp.loc[:,'mf1']=mf1_var.loc[:,col_mf1[0].to_list()[0]].copy()
            var_temp.loc[:,'mf2']=mf2_var.loc[:,col_mf2[0].to_list()[0]].copy()
            cmb_fitness[b+'_both']=gaussianMeanAndVariance(rgr_temp.loc[pheno_both.index,:].copy(),var_temp.loc[pheno_both.index,:].copy())

        if len(pheno_mf1.index)>0:
            rgr_temp=pd.DataFrame(index=mf1_RGR.index,columns=['mf1'])
            var_temp=pd.DataFrame(index=mf1_RGR.index,columns=['mf1'])
            rgr_temp.loc[:,'mf1']=mf1_RGR.loc[:,col_mf1[0].to_list()[0]].copy()
            var_temp.loc[:,'mf1']=mf1_var.loc[:,col_mf1[0].to_list()[0]].copy()
            cmb_fitness[b+'_mf1']=gaussianMeanAndVariance(rgr_temp.loc[pheno_mf1.index,:].copy(),var_temp.loc[pheno_mf1.index,:].copy())
        if len(pheno_mf2.index)>0:
            rgr_temp=pd.DataFrame(index=mf1_RGR.index,columns=['mf2'])
            var_temp=pd.DataFrame(index=mf1_RGR.index,columns=['mf2'])
            rgr_temp.loc[:,'mf2']=mf2_RGR.loc[:,col_mf2[0].to_list()[0]].copy()
            var_temp.loc[:,'mf2']=mf2_var.loc[:,col_mf2[0].to_list()[0]].copy()
            cmb_fitness[b+'_mf2']=gaussianMeanAndVariance(rgr_temp.loc[pheno_mf2.index,:].copy(),var_temp.loc[pheno_mf2.index,:].copy())
            ## calculate combined fitness

        cmb_fit_final=combine_both_feeds(cmb_fit_final,cmb_fitness,b)

    pheno_call_df2=applyPhenocall_CI(cmb_fit_final,lower_cut=np.log2(0.45),upper_cut=np.log2(2.05))

    # for b in backgrounds:
    #     pheno_call_df2.loc[pheno_call_df.index,b+'_relative_filter']=pheno_call_df.loc[pheno_call_df.index,b+'_relative_filter'].copy()
    #     pheno_call_df2.loc[pheno_call_df.index,b+'_feed']=pheno_call_df.loc[pheno_call_df.index,b+'_feed'].copy()
    pheno_call_df2=apply_filter_testInput(pheno_call_df2,mean_df_d0_mf1,mean_df_d0_mf2,rel_cut=-12,sex=backgrounds)
    return pheno_call_df2

def combine_both_feeds(cmb_fit_final,cmb_fitness,b):
    try:
        mn_list=[]
        sd_list=[]
        var_list=[]
        for k in  cmb_fitness.keys():
            mn_list.append(cmb_fitness[k][0])
            sd_list.append(cmb_fitness[k][1])
            var_list.append(cmb_fitness[k][2])
        if len(mn_list)>0:
            mn=pd.concat(mn_list)
            sd=pd.concat(sd_list)
            var=pd.concat(var_list)
            cmb_fit_final[b]=[mn,sd,var]
    except:
        import pdb; pdb.set_trace()
    return cmb_fit_final




def singleMeanAndVariance(rel_fit,rel_err):
    mean_df=rel_fit.iloc[:,0]
    var_max=rel_err.iloc[:,0]
    sd_max=var_max.apply(np.sqrt)

    return mean_df,sd_max,var_max


def applyPhenocall_CI(fit_df,lower_cut=-1,upper_cut=1,cut_para={'GCKO2':{'lower':-1.6,'upper':1},'g145480':{'lower':-1,'upper':1}}):
    ''' This function is used for define the pheno_call '''
    #
    print('phenocall_test')
    for k in fit_df.keys():
        key1=k
        break

    viz_df=pd.DataFrame(index=fit_df[key1][0].index)

    for key,item in fit_df.items():
        # we will calculate p-vlaue and z score
        if k =='GCKO2':
            lower_cut=cut_para['GCKO2']['lower']
        else:
            lower_cut=cut_para['g145480']['lower']

        m=item[0]
        s=item[1]
        diff_max=m+2*s
        diff_min=m-2*s
        ##
        viz_df[key+'_RGR'] = m
        viz_df[key+'_sd'] = s
        viz_df[key+'_var'] = item[2]
        viz_df[key+'_diff_max'] = m+2*s
        viz_df[key+'_diff_min'] = m-2*s

        viz_df[key+'_pheno'] = 'No Power'
        viz_df[key+'_relative_filter'] = 'Not applied'
        viz_df[key+'_feed'] = 'Mf1 and Mf2'

        red_df=viz_df[(viz_df[key+'_diff_max'] < lower_cut) & (viz_df[key+'_diff_min'] < lower_cut)]

        # not_red_df=viz_df[(viz_df[key+'_diff_max'] < upper_cut) & (viz_df[key+'_diff_min'] > lower_cut)]
        not_red_df=viz_df[(viz_df[key+'_diff_min'] > lower_cut)]
        # increased_df=viz_df[(viz_df[key+'_diff_max'] > upper_cut) & (viz_df[key+'_diff_min'] > upper_cut)]

        viz_df.loc[red_df.index,key+'_pheno'] = 'Reduced'
        viz_df.loc[not_red_df.index,key+'_pheno'] = 'Not Reduced'
        # viz_df.loc[increased_df.index,key+'_pheno'] = 'Increased'
    return viz_df


def propagated_relative_baundance_plot(mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2,mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2,plot_info):
     '''  We are going to plot propagated relative abundance plot  '''

     if 'rel_file' in plot_info.keys():
         print('plotting propagated relative abundance')
         geneConv=plot_info['geneConv']
         #plot_propgated_relative_abunndance(mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2,mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2,geneConv,plot_info)

         ## plot selected genes
         # selected_genes=['PBANKA_1024600','PBANKA_0501200','PBANKA_0101100','PBANKA_1422100','PBANKA_0616700','PBANKA_1359700', 'PBANKA_1359600','PBANKA_0510600']
         # plot_propgated_relative_abunndance(mean_df_d0_mf1.loc[selected_genes,:],var_df_d0_mf1.loc[selected_genes,:],mean_df_d0_mf2.loc[selected_genes,:],var_df_d0_mf2.loc[selected_genes,:],mean_df_d13_mf1.loc[selected_genes,:],var_df_d13_mf1.loc[selected_genes,:],mean_df_d13_mf2.loc[selected_genes,:],var_df_d13_mf2.loc[selected_genes,:],geneConv,plot_info)
         #
         #





def calculate_RGR(m_df_d0,var_df_d0,m_df_d13,var_df_d13,control_genes):
    ''' computing relative growth rate by divideing day13/day0'''
    # all relative abundance are on log scale
    m_df_d13.columns=m_df_d13.columns.str.replace('d13_','')
    var_df_d13.columns=var_df_d13.columns.str.replace('d13_','')
    m_df_d0.columns=m_df_d0.columns.str.replace('d0_','')
    var_df_d0.columns=var_df_d0.columns.str.replace('d0_','')
    rel_fitness=m_df_d13-m_df_d0
    rel_var=var_df_d13+var_df_d0

    control_gene_info={}
    for col in rel_fitness.columns:

        control_fitness=rel_fitness.loc[control_genes,[col]].copy()
        control_var=rel_var.loc[control_genes,[col]].copy()
        l=gaussianMeanAndVariance(control_fitness.T,control_var.T)
        control_gene_info[col]=l # l[0] mean l[1]  SD   l[2] variance

    #     print ('control_fitness', col, control_fitness[abs(control_fitness)[col]>0.75])
    #

    print(control_gene_info)
    #  we want to normalize by control genes
    normalized_fit=rel_fitness.copy()
    normalized_var=rel_var.copy()

    for col in rel_fitness.columns:
        ctr_mean=control_gene_info[col][0][col]  #  0 mean
        ctr_var=control_gene_info[col][2][col]  # 2 variance

        # this is the relative mean
        normalized_fit.loc[:,col]=rel_fitness.loc[:,col].sub(ctr_mean)
        # relative variance on log scale

        normalized_var.loc[:,col]=rel_var.loc[:,col].add(ctr_var)

    return normalized_fit,normalized_var











def plot_each_step_mean_var(mean_df_d0_mf1,var_df_d0_mf1,mean_df_d0_mf2,var_df_d0_mf2,mean_df_d13_mf1,var_df_d13_mf1,mean_df_d13_mf2,var_df_d13_mf2):
    ## compute relative ratio betwen day13 and day0
    trace_d0_female_mf1 = go.Scatter(
        x = mean_df_d0_mf1['GCKO2_d0_NA_NA'],
        y = var_df_d0_mf1['GCKO2_d0_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d0_mf1.shape[0],
        opacity=0.7,
        text=mean_df_d0_mf1.index)
    trace_d0_female_mf2 = go.Scatter(
        x = mean_df_d0_mf2['GCKO2_d0_NA_NA'],
        y = var_df_d0_mf2['GCKO2_d0_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d0_mf2.shape[0],
        opacity=0.7,
        text=mean_df_d0_mf2.index)

    trace_d0_male_mf1 = go.Scatter(
        x = mean_df_d0_mf1['g145480_d0_NA_NA'],
        y = var_df_d0_mf1['g145480_d0_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d0_mf1.shape[0],
        opacity=0.7,
        text=mean_df_d0_mf1.index)

    trace_d0_male_mf2 = go.Scatter(
        x = mean_df_d0_mf2['g145480_d0_NA_NA'],
        y = var_df_d0_mf2['g145480_d0_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d0_mf2.shape[0],
        opacity=0.7,
        text=mean_df_d0_mf2.index)

    trace_d13_female_mf1 = go.Scatter(
        x = mean_df_d13_mf1['GCKO2_d13_NA_NA'],
        y = var_df_d13_mf1['GCKO2_d13_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d13_mf1.shape[0],
        opacity=0.7,
        text=mean_df_d13_mf1.index)

    trace_d13_female_mf2 = go.Scatter(
        x = mean_df_d13_mf2['GCKO2_d13_NA_NA'],
        y = var_df_d13_mf2['GCKO2_d13_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d13_mf2.shape[0],
        opacity=0.7,
        text=mean_df_d13_mf2.index)

    trace_d13_male_mf1 = go.Scatter(
        x = mean_df_d13_mf1['g145480_d13_NA_NA'],
        y = var_df_d13_mf1['g145480_d13_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d13_mf1.shape[0],
        opacity=0.7,
        text=mean_df_d13_mf1.index)

    trace_d13_male_mf2 = go.Scatter(
        x = mean_df_d13_mf2['g145480_d13_NA_NA'],
        y = var_df_d13_mf2['g145480_d13_NA_NA'],
        mode = 'markers',
        marker=dict(size=5,color='red'),
        name='(# of markers=%d)'%mean_df_d13_mf2.shape[0],
        opacity=0.7,
        text=mean_df_d13_mf2.index)

    ### distributions
    x=mean_df_d0_mf1['GCKO2_d0_NA_NA'].values
    x_grid=np.linspace(x.min(), x.max(), 500)
    trace_d0_mf1_dist_female = go.Histogram( x=x,
    name='dist',
    marker_color='#bdbdbd',
    opacity=0.75,
    histnorm='probability')

    trace_d0_mf1_kde_female= go.Scatter(x=x_grid, y=kde_sklearn(x, x_grid, bandwidth=0.8),
                    mode='lines',name='kde')

    ### distributions
    x=mean_df_d0_mf2['GCKO2_d0_NA_NA'].values
    x_grid=np.linspace(x.min(), x.max(), 500)
    trace_d0_mf2_dist_female = go.Histogram( x=x,
    name='dist',
    marker_color='#bdbdbd',
    opacity=0.75,
    histnorm='probability')

    trace_d0_mf2_kde_female= go.Scatter(x=x_grid, y=kde_sklearn(x, x_grid, bandwidth=0.8),
                    mode='lines',name='kde')
    ### distributions
    x=mean_df_d0_mf1['g145480_d0_NA_NA'].values
    x_grid=np.linspace(x.min(), x.max(), 500)
    trace_d0_mf1_dist_male = go.Histogram( x=x,
    name='dist',
    marker_color='#bdbdbd',
    opacity=0.75,
    histnorm='probability')

    trace_d0_mf1_kde_male= go.Scatter(x=x_grid, y=kde_sklearn(x, x_grid, bandwidth=0.8),
                    mode='lines',name='kde')
    ### distributions
    x=mean_df_d0_mf2['g145480_d0_NA_NA'].values
    x_grid=np.linspace(x.min(), x.max(), 500)
    trace_d0_mf2_dist_male = go.Histogram( x=x,
    name='dist',
    marker_color='#bdbdbd',
    opacity=0.75,
    histnorm='probability')

    trace_d0_mf2_kde_male= go.Scatter(x=x_grid, y=kde_sklearn(x, x_grid, bandwidth=0.8),
                    mode='lines',name='kde')


    ### distributions
    x=mean_df_d13_mf1['GCKO2_d13_NA_NA'].values
    x_grid=np.linspace(x.min(), x.max(), 500)
    trace_d13_mf1_dist_female = go.Histogram( x=x,
    name='dist',
    marker_color='#bdbdbd',
    opacity=0.75,
    histnorm='probability')

    trace_d13_mf1_kde_female= go.Scatter(x=x_grid, y=kde_sklearn(x, x_grid, bandwidth=0.8),
                    mode='lines',name='kde')

    x=mean_df_d13_mf2['GCKO2_d13_NA_NA'].values
    x_grid=np.linspace(x.min(), x.max(), 500)
    trace_d13_mf2_dist_female = go.Histogram( x=x,
    name='dist',
    marker_color='#bdbdbd',
    opacity=0.75,
    histnorm='probability')

    trace_d13_mf2_kde_female= go.Scatter(x=x_grid, y=kde_sklearn(x, x_grid, bandwidth=0.8),
                    mode='lines',name='kde')


    ### distributions
    x=mean_df_d13_mf1['g145480_d13_NA_NA'].values
    x_grid=np.linspace(x.min(), x.max(), 500)
    trace_d13_mf1_dist_male = go.Histogram( x=x,
    name='dist',
    marker_color='#bdbdbd',
    opacity=0.75,
    histnorm='probability')

    trace_d13_mf1_kde_male= go.Scatter(x=x_grid, y=kde_sklearn(x, x_grid, bandwidth=0.8),
                    mode='lines',name='kde')

    x=mean_df_d13_mf2['g145480_d13_NA_NA'].values
    x_grid=np.linspace(x.min(), x.max(), 500)
    trace_d13_mf2_dist_male = go.Histogram( x=x,
    name='dist',
    marker_color='#bdbdbd',
    opacity=0.75,
    histnorm='probability')

    trace_d13_mf2_kde_male= go.Scatter(x=x_grid, y=kde_sklearn(x, x_grid, bandwidth=0.8),
                    mode='lines',name='kde')





    # fig, ax = plt.subplots()
    # for bandwidth in [0.8,0.9,1.0]:
    #     ax.plot(x_grid, kde_sklearn(x, x_grid, bandwidth=bandwidth),
    #         label='bandwidth={0}'.format(bandwidth), linewidth=3, alpha=0.5)
    # ax.hist(x, 50, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
    # ax.set_xlim(-1, 32)
    # ax.legend(loc='upper left')
    # # plt.ylim(0, 0.25)
    # plt.xlabel('time')
    # plt.ylabel('distributions of genes that have steepest slope')
    # plt.savefig('/Users/vikash/Documents/Projects/GeneRegulatory/Final_analysis/io/SOMap/figures/wave_continuous_sigmoidal_screen.pdf')

    fig = make_subplots(rows=4, cols=4,subplot_titles=("Input error GCKO2(day0) mf1", "Input error 145480(day0) mf1",
    "Distribution GCKO2(day0) mf1", "Distribution 145480(day0) mf1","Input error GCKO2(day0) mf2",
    "Distribution 145480(day0) mf2", "Distribution GCKO2(day0) mf2","Input error 145480(day0) mf2",
    "Output error GCKO2(day13) mf1","Output error 145480(day13) mf1","Distribution GCKO2(day13) mf1",
    "Distribution 145480(day13) mf1","Output error GCKO2(day13) mf2","Output error 145480(day13) mf2","Distribution GCKO2(day13_error_mean) mf2",
    "Distribution 145480(day13) mf2"))

    fig.append_trace(trace_d0_female_mf1,row=1, col=1)
    fig.append_trace(trace_d0_male_mf1,row=1, col=2)
    fig.append_trace(trace_d0_mf1_dist_female,row=1, col=3)
    fig.append_trace(trace_d0_mf1_kde_female,row=1, col=3)
    fig.append_trace(trace_d0_mf1_dist_male,row=1, col=4)
    fig.append_trace(trace_d0_mf1_kde_male,row=1, col=4)


    fig.append_trace(trace_d0_female_mf2,row=2, col=1)
    fig.append_trace(trace_d0_male_mf2,row=2, col=2)
    fig.append_trace(trace_d0_mf2_dist_female,row=2, col=3)
    fig.append_trace(trace_d0_mf2_kde_female,row=2, col=3)
    fig.append_trace(trace_d0_mf2_dist_male,row=2, col=4)
    fig.append_trace(trace_d0_mf2_kde_male,row=2, col=4)



    fig.append_trace(trace_d13_female_mf1,row=3, col=1)
    fig.append_trace(trace_d13_male_mf1,row=3, col=2)
    fig.append_trace(trace_d13_mf1_dist_female,row=3, col=3)
    fig.append_trace(trace_d13_mf1_kde_female,row=3, col=3)
    fig.append_trace(trace_d13_mf1_dist_male,row=3, col=4)
    fig.append_trace(trace_d13_mf1_kde_male,row=3, col=4)

    fig.append_trace(trace_d13_female_mf2,row=4, col=1)
    fig.append_trace(trace_d13_male_mf2,row=4, col=2)
    fig.append_trace(trace_d13_mf2_dist_female,row=4, col=3)
    fig.append_trace(trace_d13_mf2_kde_female,row=4, col=3)
    fig.append_trace(trace_d13_mf2_dist_male,row=4, col=4)
    fig.append_trace(trace_d13_mf2_kde_male,row=4, col=4)

    # Update xaxis properties
    # fig.update_yaxes(title_text="relative error",row=1, col=1,range=[0,0.8])
    # for pool6
    fig.update_yaxes(title_text="relative error",row=1, col=1,range=[0,1.25])
    fig.update_yaxes(title_text="relative error ",row=2, col=1,range=[0,0.8])
    fig.update_yaxes(title_text="relative error "    ,row=3, col=1,range=[0,0.8])
    fig.update_yaxes(title_text="relative error "    ,row=4, col=1,range=[0,0.8])
    fig.update_yaxes(title_text="relative error",row=1, col=2,range=[0,0.8])
    fig.update_yaxes(title_text="relative error ",row=2, col=2,range=[0,0.8])
    fig.update_yaxes(title_text="relative error "    ,row=3, col=2,range=[0,0.8])
    fig.update_yaxes(title_text="relative error "    ,row=4, col=2,range=[0,0.8])

    fig.update_yaxes(title_text="probability"    ,row=1, col=3)
    fig.update_yaxes(title_text="probability"    ,row=2, col=3)
    fig.update_yaxes(title_text="probability"    ,row=3, col=3)
    fig.update_yaxes(title_text="probability"    ,row=4, col=3)
    fig.update_yaxes(title_text="probability"    ,row=1, col=4)
    fig.update_yaxes(title_text="probability"    ,row=2, col=4)
    fig.update_yaxes(title_text="probability"    ,row=3, col=4)
    fig.update_yaxes(title_text="probability"    ,row=4, col=4)
    # Update yaxis properties
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=1, col=1,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=2, col=1,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=3, col=1,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=4, col=1,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=1, col=2,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=2, col=2,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=3, col=2,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=4, col=2,range=[-18,0])

    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=1, col=3,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=2, col=3,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=3, col=3,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=4, col=3,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=1, col=4,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=2, col=4,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=3, col=4,range=[-18,0])
    fig.update_xaxes(title_text="log2 (relative abundance or count)", row=4, col=4,range=[-18,0])


    fig.update_layout(height=2000, width=2000, title_text="Diffrent steps of input/output error analysis")
    fig.show()



def plot_relative_abunndance_daywise(sex_list,geneConv,plot_info):
    ''' We will plot genes sex-specific wise '''

    #### input ####
    # sex_list: sex_list[0]=mean and sex_list[1]=SD
    # geneConv: when we want to give name of gene
    # out_pdf: will generated pdf file
    ########
    out_pdf=plot_info['pdf']
    mf=plot_info['mf']
    day=plot_info['d']
    sex=plot_info['sex']
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    n=2
    genes=sex_list[0].index;

    labels=[]
    for gene in genes:
        if gene in geneConv.keys():
            labels.append(geneConv[gene])
        else:
            labels.append(gene)

    ### get number of subplots

    num_subplots=2*len(genes) ### number of subplots

    l=len(genes)
    loops=int(l/(n*n))
    rem=l % (n*n)
    k=0

    per_page=num_subplots/16

    for loop in range(loops):
        fig = plt.figure(figsize=(15,15))
        for i in range(1,(n*n)+1):

            plt.subplot(n, n, i)
            x=[1,2]


            ### mf1 female
            y=[sex_list[0].loc[genes[k],sex[0]+'_'+day[0]+'_'+mf[0]],sex_list[0].loc[genes[k],sex[0]+'_'+day[1]+'_'+mf[0]]]
            yerr=[sex_list[1].loc[genes[k],sex[0]+'_'+day[0]+'_'+mf[0]],sex_list[1].loc[genes[k],sex[0]+'_'+day[1]+'_'+mf[0]]]
            plt.errorbar(x,y, yerr=yerr, fmt='b--')

            ### mf2 female
            y=[sex_list[0].loc[genes[k],sex[0]+'_'+day[0]+'_'+mf[1]],sex_list[0].loc[genes[k],sex[0]+'_'+day[1]+'_'+mf[1]]]
            yerr=[sex_list[1].loc[genes[k],sex[0]+'_'+day[0]+'_'+mf[1]],sex_list[1].loc[genes[k],sex[0]+'_'+day[1]+'_'+mf[1]]]
            plt.errorbar(x,y, yerr=yerr, fmt='r--')

            ### mf1 male
            y=[sex_list[0].loc[genes[k],sex[1]+'_'+day[0]+'_'+mf[0]],sex_list[0].loc[genes[k],sex[1]+'_'+day[1]+'_'+mf[0]]]
            yerr=[sex_list[1].loc[genes[k],sex[1]+'_'+day[0]+'_'+mf[0]],sex_list[1].loc[genes[k],sex[1]+'_'+day[1]+'_'+mf[0]]]
            plt.errorbar(x,y, yerr=yerr, fmt='b-')

            ### mf2 male
            y=[sex_list[0].loc[genes[k],sex[1]+'_'+day[0]+'_'+mf[1]],sex_list[0].loc[genes[k],sex[1]+'_'+day[1]+'_'+mf[1]]]
            yerr=[sex_list[1].loc[genes[k],sex[1]+'_'+day[0]+'_'+mf[1]],sex_list[1].loc[genes[k],sex[1]+'_'+day[1]+'_'+mf[1]]]
            plt.errorbar(x,y, yerr=yerr, fmt='r-')


            plt.ylabel('log2 relative fitness')
            plt.title(labels[k])
            plt.legend(('mf1_GCKO2', 'mf2_GCKO2','mf1_145480', 'mf2_145480'))
            plt.xticks([1, 1.5, 2],['day0', '', 'day13'],fontsize=15)

            plt.ylim(-18, 1)

            k=k+1
        pdf.savefig(fig)

    ## for the remaing one

    fig = plt.figure(figsize=(15,15))
    for i in range(1,rem+1):



        plt.subplot(n, n, i)
        x=[1,2]


        ### mf1 female
        y=[sex_list[0].loc[genes[k],sex[0]+'_'+day[0]+'_'+mf[0]],sex_list[0].loc[genes[k],sex[0]+'_'+day[1]+'_'+mf[0]]]
        yerr=[sex_list[1].loc[genes[k],sex[0]+'_'+day[0]+'_'+mf[0]],sex_list[1].loc[genes[k],sex[0]+'_'+day[1]+'_'+mf[0]]]
        plt.errorbar(x,y, yerr=yerr, fmt='b--')

        ### mf2 female
        y=[sex_list[0].loc[genes[k],sex[0]+'_'+day[0]+'_'+mf[1]],sex_list[0].loc[genes[k],sex[0]+'_'+day[1]+'_'+mf[1]]]
        yerr=[sex_list[1].loc[genes[k],sex[0]+'_'+day[0]+'_'+mf[1]],sex_list[1].loc[genes[k],sex[0]+'_'+day[1]+'_'+mf[1]]]
        plt.errorbar(x,y, yerr=yerr, fmt='r--')

        ### mf1 male
        y=[sex_list[0].loc[genes[k],sex[1]+'_'+day[0]+'_'+mf[0]],sex_list[0].loc[genes[k],sex[1]+'_'+day[1]+'_'+mf[0]]]
        yerr=[sex_list[1].loc[genes[k],sex[1]+'_'+day[0]+'_'+mf[0]],sex_list[1].loc[genes[k],sex[1]+'_'+day[1]+'_'+mf[0]]]
        plt.errorbar(x,y, yerr=yerr, fmt='b-')

        ### mf2 male
        y=[sex_list[0].loc[genes[k],sex[1]+'_'+day[0]+'_'+mf[1]],sex_list[0].loc[genes[k],sex[1]+'_'+day[1]+'_'+mf[1]]]
        yerr=[sex_list[1].loc[genes[k],sex[1]+'_'+day[0]+'_'+mf[1]],sex_list[1].loc[genes[k],sex[1]+'_'+day[1]+'_'+mf[1]]]
        plt.errorbar(x,y, yerr=yerr, fmt='r-')



        plt.ylabel('log2 relative fitness')
        plt.title(labels[k])
        plt.legend(('mf1_GCKO2', 'mf2_GCKO2','mf1_g145480', 'mf2_g145480'))
        plt.xticks([1, 1.5, 2],['day0', '', 'day13'],fontsize=15)

        plt.ylim(-18, 1)

        k=k+1
        pdf.savefig(fig)
    pdf.close()


def getNewIdfromPrevID(pgenes,prev_to_new,db_df):

    ''' this is used for gene id conversion '''

    not_found=[]
    geneConv={}  ### old Id to description
    geneConv_new={} ### new Id to description
    old_to_new_ids={} ### old_id  to new_id


    for item in pgenes:
        if item in prev_to_new.keys():
            tmp= db_df[db_df['Gene ID']==prev_to_new[item]].copy()
            if tmp.empty:
                not_found.append(item)
                geneConv[item]=item
                old_to_new_ids[item]=item
                geneConv_new[item]=item
            else:

                if tmp['Gene Name or Symbol'].to_list()[0]=='NA':
                    geneConv[item]=tmp['Product Description'].to_list()[0]+'|'+prev_to_new[item]
                    old_to_new_ids[item]=prev_to_new[item]
                    geneConv_new[prev_to_new[item]]=tmp['Product Description'].to_list()[0]+'|'+prev_to_new[item]
                else:
                    geneConv[item]=tmp['Product Description'].to_list()[0]+'|'+tmp['Gene Name or Symbol'].to_list()[0]+'|'+prev_to_new[item]
                    old_to_new_ids[item]=prev_to_new[item]
                    geneConv_new[prev_to_new[item]]=tmp['Product Description'].to_list()[0]+'|'+tmp['Gene Name or Symbol'].to_list()[0]+'|'+prev_to_new[item]
    print('not found genes new ids for old gene ids ', not_found)
    return geneConv,old_to_new_ids,geneConv_new




def find_mf1_columns(mossifeed_df,saveColumns,save_df):

    # find mosquito feed diffrence mf1 and mf2 mosquito feed

    mossifeed_flag = mossifeed_df.columns.str.extract("_(.+mf[1])+")
    mossifeed_columns = saveColumns[np.where(~ mossifeed_flag.isnull())[0]]
    mossifeed_df = save_df.loc[:, mossifeed_columns]

    #

    mossifeed_df.columns= mossifeed_df.columns.str.replace("PbFertility1_[0-9]+_", '', regex=True)
    mossifeed_df.columns = mossifeed_df.columns.str.replace("_r[12]_R[12]$", '', regex=True)
    # colNames1=mossifeed_df.columns.str.replace("_R[12]$", '',regex=True)
    # mossifeed_df.columns=colNames1
    # colNames2=mossifeed_df.columns.str.replace("_r[12]", '',regex=True)
    # mossifeed_df.columns=colNames2
    colNames4=mossifeed_df.columns.str.replace("\\.[12]$", '', regex=True)
    mossifeed_df.columns=colNames4
    colNames5=mossifeed_df.columns.str.replace("\\_mf[1]$", '', regex=True)
    mossifeed_df.columns=colNames5

    map_mossifeed = getGroupMap(mossifeed_columns, mossifeed_df.columns)
    return map_mossifeed



def find_mf2_columns(mossifeed_df,saveColumns,save_df):

    # find mosquito feed diffrence mf1 and mf2 mosquito feed

    mossifeed_flag = mossifeed_df.columns.str.extract("_(.+mf[2])+")
    mossifeed_columns = saveColumns[np.where(~ mossifeed_flag.isnull())[0]]
    mossifeed_df = save_df.loc[:, mossifeed_columns]

    #

    mossifeed_df.columns= mossifeed_df.columns.str.replace("PbFertility1_[0-9]+_", '', regex=True)
    mossifeed_df.columns = mossifeed_df.columns.str.replace("_r[12]_R[12]$", '', regex=True)
    # colNames1=mossifeed_df.columns.str.replace("_R[12]$", '',regex=True)
    # mossifeed_df.columns=colNames1
    # colNames2=mossifeed_df.columns.str.replace("_r[12]", '',regex=True)
    # mossifeed_df.columns=colNames2
    colNames4=mossifeed_df.columns.str.replace("\\.[12]$", '', regex=True)
    mossifeed_df.columns=colNames4
    colNames5=mossifeed_df.columns.str.replace("\\_mf[2]$", '', regex=True)
    mossifeed_df.columns=colNames5

    map_mossifeed = getGroupMap(mossifeed_columns, mossifeed_df.columns)
    return map_mossifeed



def readInputandCountFile(df):
    input=pd.read_csv('/Users/vikash/Documents/Projects/Claire/Fertility_screen/DataFiles/input_vector.txt', sep='\t')

    ### systamatic tests
    # 1) gneral view of contamination we need to find not common genes
    contamin_genes=set(df.index.to_list())-set(input['gene_id'].to_list())
    if len(contamin_genes)>0:

        conta_df=df.loc[contamin_genes,:].copy()
        print('Contaminate genes are')

    else:
        print('There is no contaminate genes ')
    ## identify drop out genes
    # find at vitro level
    tmp=df.copy()
    saveColumns=tmp.columns
    vitro_flag = tmp.columns.str.extract("_(t[12])+")
    vitro_columns = saveColumns[np.where(~ vitro_flag.isnull())[0]]

    drop_df1= df.loc[:,vitro_columns].copy()
    drop_df1.loc[:,'row_mean']=drop_df1.sum(axis=1)
    ## required columns
    dp_genes1= drop_df1.index[drop_df1['row_mean']<50].to_list()[:-1]       ####

    ####  identify drop outs at input level (d0)

    input_flag = tmp.columns.str.extract("_(d[0])+")
    input_columns = saveColumns[np.where(~ input_flag.isnull())[0]]

    drop_df2= df.loc[:,input_columns].copy()
    drop_df2.loc[:,'row_mean']=drop_df2.sum(axis=1)
    ## required columns
    dp_genes2= drop_df2.index[drop_df2['row_mean']<50].to_list()[:-1]       ####

    ### real input genes which are not drop out

    real_input_genes=set(input['gene_id'].to_list())-(set(dp_genes2)|set(dp_genes1))

    ### drop out real
    drop_out=real_input_genes-set(df.index.to_list())
    real_input_genes=real_input_genes-drop_out

    req_df=df.loc[real_input_genes,:].copy() ### This is the require dataframes

    return req_df


def calRel(df):

    df=df.div(df.sum(axis=1), axis=0)

    df=np.log2(df)

    return df



def cal_mean_and_sd_groupby_columns(df_modi,mapping):
    tmp_mean_T = pd.DataFrame(index=df_modi.index)
    tmp_std_T = pd.DataFrame(index=df_modi.index)
    for k,item in mapping.items():

        tmp_mean_T[k]=df_modi[item].mean(axis=1).copy()
        tmp_std_T[k]=df_modi[item].std(axis=1).copy()
    return tmp_mean_T,tmp_std_T


def getColumnsFormDF(df,tag):
    t_cols=[]
    for t in tag:
        flag = df.columns.str.contains(t)
        cols= df.columns[np.where(flag)[0]]
        t_cols.append(cols)
    return t_cols

def calDissectionRatio(df_m,df_sd,time,time2,mf,backgrounds):
    # based on pcr mean and sd we will calculate ratios
    # df_sd = df_sd / df_m  # this is just to take only mutants
    # time= ['d0', 'd7', 'd14']
    # mf=['mf1','mf2']
    # backgrounds=['Cl15cy1STM','GCKO#2GOMOSTM','145480GOMOSTM']


    ratioDict={}
    for b in backgrounds:
        for t in time:
            for f in mf:
                string=b+'_'+t+'_'+f
                time_cols = getColumnsFormDF(df_m, [string])

                ratioDict[(b,t,f)]=time_cols[0]

    time = time2
    rel_fit=pd.DataFrame(index=df_m.index)
    # rel_fit1 = pd.DataFrame(index=df_m.index)
    rel_err = pd.DataFrame(index=df_m.index)
    # rel_err1 = pd.DataFrame(index=df_m.index) #without log scale
    for b in backgrounds:
        for t in time:
            for f in mf:
                tcol = ratioDict[(b, 'd0', f)]
                input=pd.DataFrame(index=df_m.index)
                input_err = pd.DataFrame(index=df_sd.index)
                input = pd.concat([input, df_m[tcol].copy()], axis=1)
                input_err = pd.concat([input_err, df_sd[tcol].copy()], axis=1)  # error

                tcol=ratioDict[(b, t, f)]
                tech_df = pd.DataFrame(index=df_m.index)
                tech_df_err = pd.DataFrame(index=df_sd.index)
                tech_df = pd.concat([tech_df, df_m[tcol].copy()], axis=1)
                tech_df_err = pd.concat([tech_df_err, df_sd[tcol].copy()], axis=1)  # error


                rel_fit = pd.concat([rel_fit, tech_df.iloc[:, 0:].sub(input.iloc[:, 0], axis=0)], axis=1)  # error

                # SD on log2 scale
                # tech_df_err=np.log2(tech_df_err)
                # input_err=np.log2(input_err)

                tmp=tech_df_err.pow(2, axis=1).iloc[:,0:].add(input_err.pow(2,axis=1).iloc[:,0],axis=0)
                # rel_err=pd.concat([rel_err,tmp.apply(np.sqrt)],axis=1) SD
                rel_err=pd.concat([rel_err,tmp],axis=1) # varaiance




    return rel_fit, rel_err

def gaussianMeanAndVariance(rel_fit,rel_err):
    if len(rel_fit.columns)==1:
        mean_df,sd_max,var_max=singleMeanAndVariance(rel_fit,rel_err)
    else:

        weight = rel_err.rdiv(1)

        nume_df=pd.DataFrame(rel_fit.values * weight.values, columns=rel_fit.columns, index=rel_fit.index)
        nume= nume_df.sum(axis=1)
        deno=weight.sum(axis=1)
        # calculate mean
        mean_df=nume/deno

        # calculate first varaiance
        var1= weight.sum(axis=1).rdiv(1)

        # claculate second variance
        term1=rel_fit.iloc[:, 0:].sub(mean_df, axis=0).pow(2,axis=1)
        term3=term1.div(rel_err,axis=0)
        term4=term3.sum(axis=1)
        term2=(1/(rel_fit.shape[1]-1))


        var2=var1*term4*term2

        var_max=pd.concat([var1, var2], axis=1).max(axis=1)
        sd_max=var_max.apply(np.sqrt)

    return [mean_df,sd_max,var_max]


def plotfitScatter(x,y,xlab,ylab,pdf,title):
    fig= plt.figure(figsize=(10,8))
    ax = plt.gca()
    xx=[]
    yy=[]
    pp=[]
    xpxp=[]


    for i in range(len(x)):
        xx.append(np.array(x[i]))
        yy.append(np.array(y[i]))
        z=np.polyfit(x[i], y[i], 1)
        p=np.poly1d(z)
        mini=np.min(x[i])
        maxi=np.max(x[i])
        pp.append(p)
        xp = np.linspace(mini, maxi, 100)
#         print(xp,pp)
        xpxp.append(xp)

        # # exponnetial fit
        # popt, pcov = curve_fit(exponenial_func, xx[i], yy[i], p0=(1, 1e-3, 1))
        # print(popt)
        # y3.append(exponenial_func(xp, *popt))
    lines=ax.plot(xx[0], yy[0], 'r.', xpxp[0], pp[0](xpxp[0]), 'r-',xx[1], yy[1], 'b.', xpxp[1], pp[1](xpxp[1]), 'b-',linewidth=3)
    # lines=ax.plot(xx[0], yy[0], 'r.', xpxp[0], y3[0], 'r-',xx[1], yy[1], 'b.', xpxp[1], y3[1], 'b-',xx[2], yy[2], 'g.', xpxp[2], y3[2], 'g-')
#     lines=ax.plot( xpxp[0], pp[0](xpxp[0]), 'r-', xpxp[1], pp[1](xpxp[1]), 'b-', xpxp[2], pp[2](xpxp[2]), 'g-')
    plt.xlabel(xlab,fontsize=15)
    plt.ylabel(ylab,fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(lines,['d0','d0','d13','d13'],fontsize=10);
    plt.title(title)
#     plt.legend(lines,['d0','d7','d14'],fontsize=15);
#     plt.xlim([-14,1])
    plt.savefig(pdf)

    plt.show()


def calRel_withoutLog(df):

    n=df.shape[1]
    df=df.div(df.sum(axis=1), axis=0)
    # get log value
    # cut=1e-10
    # df[df<cut]=cut;
    return df


def plotfitScatterDay(x,y,xlab,ylab,pdf,title):
    fig= plt.figure(figsize=(10,8))
    ax = plt.gca()
    xx=[]
    yy=[]
    pp=[]
    xpxp=[]
    y3=[]
    for i in range(len(x)):
        xx.append(np.array(x[i]))
        yy.append(np.array(y[i]))
        z=np.polyfit(x[i], y[i], 1)
        p=np.poly1d(z)
        mini=np.min(x[i])
        maxi=np.max(x[i])
        pp.append(p)
        xp = np.linspace(mini, maxi, 100)
#         print(xp,pp)
        xpxp.append(xp)

        # # exponnetial fit
        # popt, pcov = curve_fit(exponenial_func, xx[i], yy[i], p0=(1, 1e-3, 1))
        # print(popt)
        # y3.append(exponenial_func(xp, *popt))
    lines=ax.plot(xx[0], yy[0], 'b.', xpxp[0], pp[0](xpxp[0]),linewidth=3)
    # lines=ax.plot(xx[0], yy[0], 'r.', xpxp[0], y3[0], 'r-',xx[1], yy[1], 'b.', xpxp[1], y3[1], 'b-',xx[2], yy[2], 'g.', xpxp[2], y3[2], 'g-')
#     lines=ax.plot( xpxp[0], pp[0](xpxp[0]), 'r-', xpxp[1], pp[1](xpxp[1]), 'b-', xpxp[2], pp[2](xpxp[2]), 'g-')
    plt.xlabel(xlab,fontsize=15)
    plt.ylabel(ylab,fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(lines,['d13','d13'],fontsize=10);
    plt.title(title)
#     plt.legend(lines,['d0','d7','d14'],fontsize=15);

    plt.savefig(pdf)

    plt.show()



def calPvalZscore(cmb_fitness,geneMap):
    import scipy.stats as st

    import scipy.special as sp
    # anot, geneMap=readAnotation()
    # create df for ploting
    for k in cmb_fitness.keys():
        key1=k
        break
    comm=set(cmb_fitness[key1][0].index)

    geneName=[geneMap[item] for item in comm]
    viz_df=pd.DataFrame(index=cmb_fitness[key1][0].index)
    viz_df['geneName'] = geneName

    pval=0.01
    pval1=0.01
    upcut=np.log2(1)
    lowcut=np.log2(0.03)

    pheno_pval={}

    for key,item in cmb_fitness.items():

    # we will calculate p-vlaue and z score
        m=item[0]
        s=item[1]
        z=(upcut-m)/s
        df = pd.DataFrame(m, columns=[key])
        df[(key,'SD')]=s
        df['z_score']=z
        # df['pvalue']=1-sp.ndtr(df['z_score'])
        # df['pvalue_2tail'] = 2*(1 - sp.ndtr(df['z_score']))
        df['pvalue']=  (1 - st.norm.cdf(z))

       # we will calculate p value form 0.1
        z = (lowcut - m) / s
        df['z_score_2'] = z
        df['pvalue_2'] = (1 - st.norm.cdf(z))

        # select fast growing, NE, E, middle
        # if pvalue<0.05 and relative fitness >0
        df_fast=df[(df['pvalue'] < pval) & (df[key] > upcut)]
        df_ambiguous1=df[(df['pvalue'] < pval) & (df[key] < upcut)]
        NE_df=df[(df['pvalue'] > pval)]
        df_subset1= df[(df['pvalue_2'] >pval1)]
        df_ambiguous2 = df[(df['pvalue_2'] < pval1) & (df[key] > lowcut)]
        E_df=df[(df['pvalue_2'] < pval1)]

        # non essential genes
        NE=(NE_df.index)
        # essential genes
        E=E_df.index
        dropIdx=NE.union(E)
        # other genes
        df_other=df.drop(index=dropIdx)

        other=df_other.index

        viz_df[key[0]+'_'+key[1]+'_pheno'] = 'NA'
        viz_df[key[0]+'_'+key[1]+'_pheno'][E] = 'E'
        viz_df[key[0]+'_'+key[1]+'_pheno'][NE] = 'NE'
        viz_df[key[0]+'_'+key[1]+'_rel'] = m
        viz_df[key[0] + '_' + key[1] + '_sd'] = s

        pheno_pval['E']=E
        pheno_pval['NE'] = NE
        pheno_pval['other'] = other
        #import pdb;pdb.set_trace()  # we are checking the pvalues for normal genes .

    return viz_df



def plotMaleFemaleScatter(df):
    # fig = plt.figure(figsize=(8,8))
    fig, ax = plt.subplots(figsize=(9, 9))

    ###
    # change df_values

    df['145480_d13_pheno']=df['145480_d13_pheno'].replace({'E': 'IM', 'NE': 'FM','NA':'RM'})
    df['GCKO2_d13_pheno']=df['GCKO2_d13_pheno'].replace({'E': 'IF', 'NE': 'FF', 'NA': 'RF'})
    # viz_df['Published_cross_phenotype']=viz_df['Published_cross_phenotype'].replace({'N': 'NA'})

    # cmap={'FM':"#66c2a5", 'RM':"#8da0cb", 'IM':"#fc8d62"}
    cmap_male={'FM':"#1b9e77", 'RM':"#7570b3", 'IM':"#d95f02"}
    cmap_female={'FF':"#1b9e77", 'RF':"#7570b3", 'IF':"#d95f02"}


    for i,item in enumerate(df['GCKO2_d13_pheno'].to_list()):
        marker_style = dict(color=cmap_female[item],markersize=6, markerfacecoloralt=cmap_male[df['145480_d13_pheno'][i]],markeredgecolor='white',alpha=0.8)
        ax.plot(df['GCKO2_d13_rel'][i],df['145480_d13_rel'][i], 'o',fillstyle='left', **marker_style)


    ax.set_xlabel('Relative growth rate (Female)',fontsize=15)
    ax.set_ylabel('Relative growth rate (Male)',fontsize=15)
    ax.tick_params(axis='y',labelsize=12)
    ax.tick_params(axis='x',labelsize=12)
    ax.set_xlim(-11,3)
    ax.set_ylim(-11,3)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels)
    infertile= mpatches.Patch(color="#fc8d62", label='Infertility (F/M)')
    fertile=mpatches.Patch(color="#66c2a5", label='Normal fertility (F/M)')
    reduced_fertile=mpatches.Patch(color="#8da0cb", label='Reduced fertility (F/M)')
    plt.legend(handles=[infertile,fertile,reduced_fertile],loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3, borderaxespad=0, frameon=False, prop={'size': 6})

    fig.savefig(plot_folder + "scatter_plot_male_female_RGR_pool1.pdf")




def plotInSnsNew(viz_df):



    # sns.set_style("white")
    sns.set(style="white",font_scale=1.5)
    print('hi')

    # change df_values

    viz_df['145480_d13_pheno']=viz_df['145480_d13_pheno'].replace({'E': 'IM', 'NE': 'FM','NA':'RM'})
    viz_df['GCKO2_d13_pheno']=viz_df['GCKO2_d13_pheno'].replace({'E': 'IF', 'NE': 'FF', 'NA': 'RF'})
    # viz_df['Published_cross_phenotype']=viz_df['Published_cross_phenotype'].replace({'N': 'NA'})


    ##################### BAR PLOTS BEGINS


    plt.figure(figsize=(16*4, 20*4))
    bar_df=viz_df.sort_values(by=["145480_d13_rel"])

    # cmap={'FM':"#66c2a5", 'RM':"#8da0cb", 'IM':"#fc8d62"}
    cmap={'FM':"#1b9e77", 'RM':"#7570b3", 'IM':"#d95f02"}

    hue=[ "#1b9e77",  "#7570b3" ,"#d95f02"]
    clrs = [cmap[bar_df.loc[x,"145480_d13_pheno"]]  for x in bar_df.index]

    ax = sns.barplot(x="145480_d13_rel", y="geneName", data=bar_df, ci=None,palette = clrs)
    # new_labels = ['FM', 'RM' , 'IM']
    # for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)

    plt.errorbar(x=bar_df["145480_d13_rel"], y=ax.get_yticks(), xerr=bar_df["145480_d13_sd"], fmt='none',c='gray')

    patch =[ mpatches.Patch(color="#1b9e77", label='FM'), mpatches.Patch(color="#7570b3", label='RM'), mpatches.Patch(color="#d95f02", label='IM')]
    plt.legend(handles=patch)
    fig = ax.get_figure()
    fig.savefig(plot_folder + "bar_145480_plot_RGR_pool1.pdf")



    plt.figure(figsize=(16*4, 20*4))
    bar_df=viz_df.sort_values(by=["GCKO2_d13_rel"])

    cmap={'FF':"#1b9e77", 'RF':"#7570b3", 'IF':"#d95f02"}


    clrs = [cmap[bar_df.loc[x,"GCKO2_d13_pheno"]]  for x in bar_df.index]

    ax = sns.barplot(x="GCKO2_d13_rel", y="geneName", data=bar_df, ci=None,palette = clrs)
    plt.errorbar(x=bar_df["GCKO2_d13_rel"], y=ax.get_yticks(), xerr=bar_df["GCKO2_d13_sd"], fmt='none',c='gray')
    patch =[ mpatches.Patch(color="#1b9e77", label='FF'), mpatches.Patch(color="#7570b3", label='RF'), mpatches.Patch(color="#d95f02", label='IF')]
    plt.legend(handles=patch)
    fig = ax.get_figure()
    fig.savefig(plot_folder + "bar_GCKO2_plot_RGR_pool1.pdf")


def stepByStep_barSeqAnalysis():

    ############# STEP1 ###########

    ### we are taking gene counts which we found from barseq experiment
    ## arranged columns based on diffrent kind of errors:
    ## midgut errors, disection errors, PCR errors
    df_req, df_data,map_pcr, map_midgut, map_mossifeed,map_mf1,map_mf2 = changeHeaderFromSequencer()


    ############# STEP 2 ###########

    # we want to find out where there is contamination because due to experiment it could be some mutanats can be mixed.
    # for this we need to read input file for pools and then we need to match with what we get from barseq data

    # read Input pool file and check with data

    df_ori=readInputandCountFile(df_data)


    ### ######  Step 3  calculate relative abundance
    df_ori = df_ori + 1 ### this trick we used to remove zero count
    ref_ori = calRel(df_ori.T) # calculate relative frequency
    ref_ori = ref_ori.T # tarnspose



    ##################### plot relative log2 counts ####################

    input_genes=ref_ori.index.to_list()
    ## map previous id
    geneConv=getNewIdfromPrevID(input_genes)

    ## mf1 mean and sd
    mf1_m, mf1_sd = cal_mean_and_sd_groupby_columns(ref_ori,map_mf1)
    mf2_m, mf2_sd = cal_mean_and_sd_groupby_columns(ref_ori,map_mf2)

    # ## plot relative fitnees
    # out_pdf = plot_folder+'relative_fitness.pdf'
    # plot_relativeFitness(mf1_m, mf1_sd,mf2_m, mf2_sd,geneConv,out_pdf)

    ### end for plotting relative fitness


    ######### we are going to plot errors

    #calculate error for PCR duplicates

    # df_ori=df_ori
    # import pdb;pdb.set_trace()
    ref_ori=calRel_withoutLog(df_ori.T)
    ref_ori=ref_ori.T
    pcr_m, pcr_sd = cal_mean_and_sd_groupby_columns(ref_ori,map_pcr)  # this is the mean and difference between pcr duplicates

    midgut_m, midgut_sd = cal_mean_and_sd_groupby_columns(ref_ori,map_midgut)  # this is the mean and difference between midugut samples

    mossifeed_m, mossifeed_sd = cal_mean_and_sd_groupby_columns(ref_ori,map_mossifeed) # this is the mean and difference between mosiquito feeds

    tag=['d0_mf','d13_mf']

    ######## PCR error

    pcr_time_df,pcr_time_df_err,pcr_time_df_err1,pcr_time_df_nl,pcr_values,pcr_values_err,pcr_values_err1,pcr_values_nl=selectColumns(pcr_m, pcr_sd,tag)
    pdf=plot_folder+'PCR_error_relative abundance_pool1.pdf'

    #plotfitScatter(pcr_values,pcr_values_err,'log2(mean of relative abundances)','log2(SD of relative abundances)',pdf,'PCR_error')
    # plotfitScatter(pcr_values,pcr_values_err,'log2(mean of relative abundances)','SD of relative abundances',pdf,'PCR_error')
    # plotfitScatter(pcr_values,pcr_values_err1,'log2(mean of relative abundances)','(SD of relative abundances)/mean',pdf,'PCR_error')

    ####
    tag=['d13_mf']
    gut_time_df,gut_time_df_err,gut_time_df_err1,gut_time_df_nl,gut_values,gut_values_err,gut_values_err1,gut_values_nl=selectColumns( midgut_m, midgut_sd,tag) # find only for mutants

    pdf=plot_folder+'dissection_error_pool1.pdf'
    #plotfitScatterDay(gut_values,gut_values_err,'log2(mean relative abundances)','log2(SD of relative abundances)',pdf,'Dissection_error')
    # plotfitScatterDay(gut_values,gut_values_err1,'log2(mean relative abundances)','SD of relative abundances',pdf,'Dissection_error')


    ### we are going to plot distribution on dissection error plot

    # gut_time_df,gut_time_df_err1=plotHistogramonError(gut_values,gut_values_err1,pcr_time_df)



    #######   STEP4 calculate sctter plot male vs female


    ref_ori = calRel(df_ori.T) # calculate relative frequency
    ref_ori = ref_ori.T # tarnspose
    pcr_m, pcr_sd = cal_mean_and_sd_groupby_columns(ref_ori,map_pcr)  # this is the mean and difference between pcr duplicates

    # midgut_m, midgut_sd = cal_mean_and_sd_groupby_columns(ref_ori,map_midgut)  # this is the mean and difference between midugut samples
    #
    # mossifeed_m, mossifeed_sd = cal_mean_and_sd_groupby_columns(ref_ori,map_mossifeed) # this is the mean and difference between mosiquito feeds



    #### calculate dissection error
    time= ['d0', 'd13']
    mf=['mf1','mf2']
    backgrounds=['GCKO2','145480']
    time2= ['d13']

    rel_fitness,rel_var= calDissectionRatio(pcr_m, pcr_sd,time,time2,mf,backgrounds)

    #

    # withlogtuple=tl[0]
    # nologtuple=tl[1]
    # normalized by reference poposed by oliver
    # control_genes=['PBANKA_102460','PBANKA_100210','PBANKA_100220','PBANKA_050120']

    control_genes=['PBANKA_102460' , 'PBANKA_050120' , 'PBANKA_010110' , 'PBANKA_142210']




    control_gene_info={}
    for col in rel_fitness.columns:
        control_fitness=rel_fitness.loc[control_genes,[col]].copy()
        control_var=rel_var.loc[control_genes,[col]].copy()
        l=gaussianMeanAndVariance(control_fitness.T,control_var.T)
        control_gene_info[col]=l # l[0] mean l[1]  SD   l[2] variance

    #  we want to normalize by control genes
    normalized_fit=rel_fitness.copy()
    normalized_var=rel_var.copy()

    for col in rel_fitness.columns:
        ctr_mean=control_gene_info[col][0][col]  #  0 mean
        ctr_var=control_gene_info[col][2][col]  # 1 variance

        # this is the relative mean
        normalized_fit.loc[:,col]=rel_fitness.loc[:,col].sub(ctr_mean)
        # relative variance on log scale

        normalized_var.loc[:,col]=rel_var.loc[:,col].add(ctr_var)


    cmb_fitness={}
    time= ['d13']
    backgrounds=['GCKO2','145480']

    for b in backgrounds:
        for t in time:
            string=b+'_'+t
            tcols = getColumnsFormDF(normalized_fit, [string])
            tm=normalized_fit[tcols[0]].copy()
            terr=normalized_var[tcols[0]].copy()
            import pdb;pdb.set_trace()
            cmb_fitness[b]=gaussianMeanAndVariance(tm,terr)

    # #plotrelFitness(cmb_fitness)
    # ##########  This function  is used for preparing plots for fertility website
    # res_df=prepareForPGfertility(cmb_fitness)


    viz_df=calPvalZscore(cmb_fitness,geneConv)



    ### create sctter plot male female relative fitness
    # plotMaleFemaleScatter(viz_df)



    plotInSnsNew(viz_df)

def getValues(df):
    df_values=[]
    for t_values in df.values:
        for val in t_values:
            df_values.append(val)
    return df_values

def getColumnsFormDF(df,tag):
    t_cols=[]
    for t in tag:
        flag = df.columns.str.contains(t)
        cols= df.columns[np.where(flag)[0]]
        t_cols.append(cols)
    return t_cols

def selectColumns(df_m, df_sd,tag):
    #with this function we want to compute error for each time point
    # df_m is the mean df_sd is SD.
    # test whether there is duplicates in genes
    if len(df_m.index) !=len(df_m.index.unique()):
        print ("there is duplicate genes: We should fixed which gene do we want to take in analysis")
        exit()


    time_cols=getColumnsFormDF(df_m,tag)
    df_rel=df_sd/df_m # this is just to take only mutants

    time_df_nolog=[]
    time_df=[]
    time_df_err=[]
    time_df_err1=[]
    # technical replicates

    for tcol in time_cols:
        tech_df = pd.DataFrame(index=df_m.index)
        tech_df_err = pd.DataFrame(index=df_sd.index)
        tech_df_err1 = pd.DataFrame(index=df_rel.index)
        tech_df = pd.concat([tech_df,df_m[tcol].copy()],axis=1)
        tech_df_err = pd.concat([tech_df_err,df_sd[tcol].copy()],axis=1) #  error
        tech_df_err1 = pd.concat([tech_df_err1,df_rel[tcol].copy()],axis=1) #  error
        cut=1e-10
        if tech_df.isnull().values.any():
            print ("relative abundance contains nan value do you want to ignore? we put 1")
            tech_df=tech_df.fillna(1)
        if tech_df_err.isnull().values.any():
            print ("relative abundance error contains nan value do you want to ignore? we put 0")
            tech_df_err=tech_df_err.fillna(0)
        # import pdb;pdb.set_trace()
        # tech_df[tech_df < cut]=cut
        time_df_nolog.append(tech_df)
        tech_df=np.log2(tech_df)
        # tech_df_err[tech_df_err < cut]=cut
        # tech_df_err=np.log2(tech_df_err)

        time_df.append(tech_df)
        time_df_err.append(tech_df_err)
        time_df_err1.append(tech_df_err1)

    # collaspe in one vector calculate errors and values
    l_values=[]
    for item in time_df:
# technical replicates ratio
        values=getValues(item)
        l_values.append(values)

    l_values_no_log = []
    for item in time_df_nolog:
        values = getValues(item)
        l_values_no_log.append(values)

    l_values_err=[]
    for item in time_df_err:
# technical replicates error
        values=getValues(item)
        l_values_err.append(values)

    l_values_err1=[]
    for item in time_df_err1:
# technical replicates error
        values=getValues(item)
        l_values_err1.append(values)

    return time_df,time_df_err,time_df_err1,time_df_nolog,l_values,l_values_err,l_values_err1,l_values_no_log

def plot_relativeFitness(mf1_m, mf1_sd,mf2_m, mf2_sd,geneConv,out_pdf):
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    n=2
    genes=mf1_m.index;

    labels=[]
    for gene in genes:
        if gene in geneConv.keys():
            labels.append(geneConv[gene])
        else:
            labels.append(gene)

    ### get number of subplots

    num_subplots=2*len(genes) ### number of subplots

    l=len(genes)
    loops=int(l/(n*n))
    rem=l % (n*n)
    k=0

    per_page=num_subplots/16

    for loop in range(loops):
        fig = plt.figure(figsize=(15,15))
        for i in range(1,(n*n)+1):

            plt.subplot(n, n, i)
            x=[1,2]


            ### mf1 female
            y=[mf1_m.loc[genes[k],'GCKO2_d0'],mf1_m.loc[genes[k],'GCKO2_d13']]
            yerr=[mf1_sd.loc[genes[k],'GCKO2_d0'],mf1_sd.loc[genes[k],'GCKO2_d13']]
            plt.errorbar(x,y, yerr=yerr, fmt='b--')

            ### mf2 female
            y=[mf2_m.loc[genes[k],'GCKO2_d0'],mf2_m.loc[genes[k],'GCKO2_d13']]
            yerr=[mf2_sd.loc[genes[k],'GCKO2_d0'],mf2_sd.loc[genes[k],'GCKO2_d13']]
            plt.errorbar(x,y, yerr=yerr, fmt='r--')

            ### mf1 male
            y=[mf1_m.loc[genes[k],'145480_d0'],mf1_m.loc[genes[k],'145480_d13']]
            yerr=[mf1_sd.loc[genes[k],'145480_d0'],mf1_sd.loc[genes[k],'145480_d13']]
            plt.errorbar(x,y, yerr=yerr, fmt='b-')

            ### mf2 male
            y=[mf2_m.loc[genes[k],'145480_d0'],mf2_m.loc[genes[k],'145480_d13']]
            yerr=[mf2_sd.loc[genes[k],'145480_d0'],mf2_sd.loc[genes[k],'145480_d13']]
            plt.errorbar(x,y, yerr=yerr, fmt='r-')

            plt.ylabel('log2 relative fitness')
            plt.title(labels[k])
            plt.legend(('mf1_female', 'mf2_female','mf1_male', 'mf2_male'))
            plt.xticks([1, 1.5, 2],['day0', '', 'day13'],fontsize=15)

            plt.ylim(-18, 1)

            k=k+1
        pdf.savefig(fig)

    ## for the remaing one
    fig = plt.figure(figsize=(15,15))
    for i in range(1,rem+1):
        fig = plt.figure(figsize=(15,15))


        plt.subplot(n, n, i)
        x=[1,2]


        ### mf1 female
        y=[mf1_m.loc[genes[k],'GCKO2_d0'],mf1_m.loc[genes[k],'GCKO2_d13']]
        yerr=[mf1_sd.loc[genes[k],'GCKO2_d0'],mf1_sd.loc[genes[k],'GCKO2_d13']]
        plt.errorbar(x,y, yerr=yerr, fmt='b--')

        ### mf2 female
        y=[mf2_m.loc[genes[k],'GCKO2_d0'],mf2_m.loc[genes[k],'GCKO2_d13']]
        yerr=[mf2_sd.loc[genes[k],'GCKO2_d0'],mf2_sd.loc[genes[k],'GCKO2_d13']]
        plt.errorbar(x,y, yerr=yerr, fmt='r--')

        ### mf1 male
        y=[mf1_m.loc[genes[k],'145480_d0'],mf1_m.loc[genes[k],'145480_d13']]
        yerr=[mf1_sd.loc[genes[k],'145480_d0'],mf1_sd.loc[genes[k],'145480_d13']]
        plt.errorbar(x,y, yerr=yerr, fmt='b-')

        ### mf2 male
        y=[mf2_m.loc[genes[k],'145480_d0'],mf2_m.loc[genes[k],'145480_d13']]
        yerr=[mf2_sd.loc[genes[k],'145480_d0'],mf2_sd.loc[genes[k],'145480_d13']]
        plt.errorbar(x,y, yerr=yerr, fmt='r-')
        plt.ylabel('log2 relative fitness')
        plt.title(labels)
        plt.legend(('mf1', 'mf2'))
        plt.xticks([1, 1.5, 2],['day0', '', 'day13'],fontsize=15)

        plt.ylim(-18, 1)

        k=k+1
    pdf.savefig(fig)
    pdf.close()








def makeBarcodeCSV():

    ### we are going to make a csv file for barcoe sample

    df=pd.read_csv("/Users/vikash/Documents/Projects/Claire/Fertility_screen/DataFiles/plasmogem_all_berghei_targeting_vectors_and_barcodes_jan20.csv")

    input=pd.read_csv('/Users/vikash/Documents/Projects/Claire/Fertility_screen/DataFiles/input_vector.txt', sep='\t')
    tmp=df.loc[:,['gene_id', 'barcode'] ].copy()
    tmp=tmp.drop_duplicates()
    ### find unique barcodes

    uniq_barcodes=tmp['barcode'].unique().tolist()
    uniq_genes=tmp['gene_id'].unique().tolist()

    store_multiple=[]
    multiple_barcode=[]
    barcode_gene={}
    for gene in uniq_genes:
        xx=tmp[tmp['gene_id']==gene]
        if xx.shape[0]>1:
            multiple_barcode.append(gene)
            store_multiple.append(xx['barcode'].to_list())
        elif xx.shape[0]==1:
            barcode_gene[gene]=xx['barcode'].to_list()[0]
        else:
            print('not found')


    ## get barcode based on PlasmoDB ID



    see_genes=[]
    for item in input['PbGEM-ID'].to_list():
        if df[df['PlasmoGEM_ID']==item].empty:
            see_genes.append(item)

    ######### write files

    out=open("barcode_file.txt", 'w')
    out.write("gene\tbarcode\n")
    for k,v in barcode_gene.items():
        out.write("%s\t%s\n"%(k,v))
    for i in range(len(multiple_barcode)):
        out.write("%s\t%s\n"%(multiple_barcode[i],'|'.join(store_multiple[i])))

    ######### write files

    out=open("barcode_gene_file.csv", 'w')
    out.write("barcode,gene\n")
    for k in uniq_barcodes:
        ## get genes
        tmp=df[df['barcode']==k]
        plasmoGEM=set(tmp['PlasmoGEM_ID'].to_list())
        pbanka=set(tmp['gene_id'].to_list())

        gene='|'.join(plasmoGEM|pbanka)

        out.write("%s,%s\n"%(k,gene))


def getGroupMap(list1,list2):
    d={}
    for i,item in enumerate(list2):
        if item not in d.keys():
            d[item]=[]
        d[item].append(list1[i])
    return d


def plotHistogramonError(gut_values,gut_values_err,pcr_time_df):

    #### gut_time_df:  is mean values at day 13


    ### gut_time_df_err1: is the sd at day 13

    ####
    input_df=pcr_time_df[0] # this is the input dataframe at time day0
    input_values=getValues(input_df)

    pdf=plot_folder+'Dissection_error.svg'

    plotfitScatterDayNoLine(gut_values,gut_values_err,'log2(relative abundances)','Relative error',pdf,'Dissection_error')


    ### plot histogram
    pdf = plot_folder + 'Distribution_input(D0)_new.svg'
    plothistNew(input_values, '', 'Frequency', pdf, 50, [-17, -2])
    import pdb;pdb.set_trace()


def plothistNew(values,xlab,ylab,pdf,bins,xlim):
    ## get relative abundance
    data = np.array(values)
    # remove nans
    data = data[~np.isnan(data)]
    fig= plt.figure(figsize=(10,8))
# ax = plt.gca()
    kwargs = dict(histtype='step', alpha=1, normed=True, bins=bins,facecolor='blue', linewidth=2)

    plt.hist(data, **kwargs)
    # plt.grid(True)
    plt.xlim(xlim)
    plt.xlabel(xlab,fontsize=20)
    plt.ylabel(ylab,fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(pdf,format="svg")


def plotfitScatterDayNoLine(x,y,xlab,ylab,pdf,title):
    fig= plt.figure(figsize=(10,8))
    ax = plt.gca()
    xx=[]
    yy=[]
    pp=[]
    xpxp=[]
    y3=[]
    for i in range(len(x)):
        xx.append(np.array(x[i]))
        yy.append(np.array(y[i]))
        z=np.polyfit(x[i], y[i], 1)
        p=np.poly1d(z)
        mini=np.min(x[i])
        maxi=np.max(x[i])
        pp.append(p)
        xp = np.linspace(mini, maxi, 100)
#         print(xp,pp)
        xpxp.append(xp)

        # # exponnetial fit
        # popt, pcov = curve_fit(exponenial_func, xx[i], yy[i], p0=(1, 1e-3, 1))
        # print(popt)
        # y3.append(exponenial_func(xp, *popt))
    # lines=ax.plot(xx[0], yy[0],'gray.', xx[1], yy[1], 'balck.')
    # p1=ax.scatter(xx[0], yy[0], color='#999999', marker='.')
    p1=ax.scatter(xx[0], yy[0], color='#000000', marker='.')
    # p2=ax.scatter(xx[1], yy[1], color='#000000', marker='.')


    #lines=ax.plot(xx[0], yy[0], 'b.', xpxp[0], pp[0](xpxp[0]), 'b-',xx[1], yy[1], 'g.', xpxp[1], pp[1](xpxp[1]), 'g-',linewidth=3)
    # lines=ax.plot(xx[0], yy[0], 'r.', xpxp[0], y3[0], 'r-',xx[1], yy[1], 'b.', xpxp[1], y3[1], 'b-',xx[2], yy[2], 'g.', xpxp[2], y3[2], 'g-')
#     lines=ax.plot( xpxp[0], pp[0](xpxp[0]), 'r-', xpxp[1], pp[1](xpxp[1]), 'b-', xpxp[2], pp[2](xpxp[2]), 'g-')
    plt.xlabel(xlab,fontsize=20)
    plt.ylabel(ylab,fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.rc('xtick',labelsize=)
    # plt.rc('ytick',labelsize=15)
    plt.legend([p1],['d13'],fontsize=16);
    plt.title(title)
#     plt.legend(lines,['d0','d7','d14'],fontsize=15);
    plt.xlim([-17,-2])
    plt.savefig(pdf,format="svg")

    plt.show()
if __name__ == '__main__':

    #makeBarcodeCSV()
    stepByStep_barSeqAnalysis()
