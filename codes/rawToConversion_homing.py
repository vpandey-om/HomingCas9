import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
code=os.getcwd()
upLevel=code.replace('codes','') ####### we are going to upper level of code directory
sys.path.insert(0,upLevel+'data')
sys.path.insert(1, upLevel+'Figures')
import statsmodels.stats.multitest as mtest

# print(sys.path)
from special_fun_fertility  import preprocessing,filter_input_dropout,getNewIdfromPrevID,getCombined_mean_variance,gaussianMeanAndVariance
# input files which will be needed for analysis of cliare screen data

data_folder=sys.path[0]

## output folder where we will write figures and output files
out_folder=sys.path[1]

# ID conversion: we used plasmoDB to convert ID for P. Berghai
prev_to_new=pickle.load(open(data_folder+'/prevTonew_PBANKA.pickle','rb'))
db_df=pd.read_csv(data_folder+'/GenesByGeneModelChars_Summary.txt', sep='\t')
db_df=db_df.fillna('NA')
new_to_prev= dict((v,k) for k,v in prev_to_new.items())
## end of databse information






def calculate_RGR_metabolic_pool(m_df_d0,var_df_d0,m_df_d13,var_df_d13,control_genes):
    ''' computing relative growth rate by divideing day13/day0'''
    # all relative abundance are on log scale
    m_df_d13.columns=m_df_d13.columns.str.replace('d14_','')
    var_df_d13.columns=var_df_d13.columns.str.replace('d14_','')
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
    #return rel_fitness,rel_var



def plot_prop_rel_abun_nutrient(mean_df_d0,var_df_d0,mean_df_d13,var_df_d13,geneConv,plot_info):
    ''' We will plot genes sex-specific wise '''

    #### input ####
    # sex_list: sex_list[0]=mean and sex_list[1]=SD
    # geneConv: when we want to give name of gene
    # out_pdf: will generated pdf file
    ########
    out_pdf=plot_info['rel_file']
    # mf=plot_info['mf']
    day=plot_info['d']
    sex=plot_info['N']

    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    n=2
    genes=mean_df_d0.index;

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
            y=[mean_df_d0.loc[genes[k],sex[0]+'_'+day[0]+'_NA'],mean_df_d13.loc[genes[k],sex[0]+'_'+day[1]+'_NA']]
            yerr=[var_df_d0.loc[genes[k],sex[0]+'_'+day[0]+'_NA'],var_df_d13.loc[genes[k],sex[0]+'_'+day[1]+'_NA']]
            plt.errorbar(x,y, yerr=yerr, fmt='r--')


            # ### mf1 male
            y=[mean_df_d0.loc[genes[k],sex[1]+'_'+day[0]+'_NA'],mean_df_d13.loc[genes[k],sex[1]+'_'+day[1]+'_NA']]
            yerr=[var_df_d0.loc[genes[k],sex[1]+'_'+day[0]+'_NA'],var_df_d13.loc[genes[k],sex[1]+'_'+day[1]+'_NA']]
            plt.errorbar(x,y, yerr=yerr, fmt='b--')


            plt.ylabel('log2 relative fitness')
            plt.title(labels[k])
            plt.legend((sex[0], sex[1]))
            plt.xticks([1, 1.5, 2],['day0', '', 'day14'],fontsize=15)

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
        y=[mean_df_d0.loc[genes[k],sex[0]+'_'+day[0]+'_NA'],mean_df_d13.loc[genes[k],sex[0]+'_'+day[1]+'_NA']]
        yerr=[var_df_d0.loc[genes[k],sex[0]+'_'+day[0]+'_NA'],var_df_d13.loc[genes[k],sex[0]+'_'+day[1]+'_NA']]
        plt.errorbar(x,y, yerr=yerr, fmt='r--')


        # ### mf1 male
        y=[mean_df_d0.loc[genes[k],sex[1]+'_'+day[0]+'_NA'],mean_df_d13.loc[genes[k],sex[1]+'_'+day[1]+'_NA']]
        yerr=[var_df_d0.loc[genes[k],sex[1]+'_'+day[0]+'_NA'],var_df_d13.loc[genes[k],sex[1]+'_'+day[1]+'_NA']]
        plt.errorbar(x,y, yerr=yerr, fmt='b--')




        plt.title(labels[k])
        plt.legend((sex[0], sex[1]))
        plt.xticks([1, 1.5, 2],['day0', '', 'day14'],fontsize=15)

        plt.ylim(-18, 1)
        plt.grid(False)
        k=k+1
        pdf.savefig(fig)
    pdf.close()



def calPvalQval(m,s,cutoff=0):
    ''' pheno_call_df'''

    z=(cutoff-m)/s
    z1=abs(z)
    pvalue=  (1 - st.norm.cdf(z.tolist()))
    # pvalue2=  (1 - st.norm.cdf(z1.tolist())) *2
    fdr=mtest.multipletests(pvalue, alpha=0.05, method='fdr_bh')
    # fdr2=mtest.multipletests(pvalue2, alpha=0.05, method='fdr_bh')

    return pvalue,fdr



def offTargetPlot(df,mdf):
    ''' calculate off target analysis'''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_var=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    grp_cols=['N','d','mf','b','t']
    grp_cols.remove('mf')
    grp_cols.remove('b')
    day_pos=grp_cols.index('d')
    days=['d0']

    for k,v in mdf.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)
            tmp_mani=mdf.iloc[v].copy()

            for dr,dr_v in tmp_mani.groupby([ 'mf']).indices.items():
                dr='_'.join(dr)
                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=mean_df.copy()
            final_var[key]=var_max.copy()

    final_mean_log=np.log2(final_mean)
    final_var_log=(final_var)/((final_mean*final_mean)*((np.log(10)**2)*np.log10(2)))
    #### get for t
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_var=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])
    grp_cols=['t']
    tt=['t1','t2']

    for k,v in mdf.groupby(grp_cols).indices.items():
        if k in tt:
            tmp_mani=mdf.iloc[v].copy()
            tmp_mn[k]=df[tmp_mani.index].mean(axis=1).copy()
            tmp_var[k]=df[tmp_mani.index].var(axis=1).copy()
            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[k]=mean_df.copy()
            final_var[k]=var_max.copy()

    final_mean_log_t=np.log2(final_mean)
    final_var_log_t=(final_var)/((final_mean*final_mean)*((np.log(10)**2)*np.log10(2)))

    ### now we want to plot
    df_final_mean=pd.concat([final_mean_log,final_mean_log_t],axis=1)
    df_final_var=pd.concat([final_var_log,final_var_log_t],axis=1)
    ###
    df_final_mean.columns=df_final_mean.columns+'_mean'
    df_final_var.columns=df_final_var.columns+'_var'

    df_final=pd.concat([df_final_mean,df_final_var],axis=1)
    df_final.to_excel('/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/offtargetResults.xlsx')
    # import pdb; pdb.set_trace()

    return None




def relative_growth_rate_analysis_homing(df,manfest_df,prev_to_new,db_df,plot_info=None):
    ''' We are going to do relative growth rate analysis'''

    ## first we need to propagate error from PCR to mosquito feed

    rel_df=df.copy()
    rel_df=rel_df.drop(columns=['Gene','Barcodes'])
    rel_df= rel_df + 1
    rel_df=rel_df.div(rel_df.sum(axis=0), axis=1)
    rel_df_log=np.log2(rel_df)
    # offTargetPlot(rel_df,manfest_df)

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
    grp_cols=['N','d','mf','b','t']
    day_pos=grp_cols.index('d')

    mean_df_d0,var_df_d0=propagate_error(rel_df,manfest_df,grp_cols,day_pos,days=['d0'])

    grp_cols=['N','d','mf','b','t']
    day_pos=grp_cols.index('d')

    mean_df_d13,var_df_d13=propagate_error(rel_df,manfest_df,grp_cols,day_pos,days=['d14'])


    if 'rel_file' in plot_info.keys():
        print('plotting propagated relative abundance')
        geneConv=plot_info['geneConv']
        plot_prop_rel_abun_nutrient(mean_df_d0,var_df_d0,mean_df_d13,var_df_d13,geneConv,plot_info)

    RGR,RGR_var=calculate_RGR_metabolic_pool(mean_df_d0.copy(),var_df_d0.copy(),mean_df_d13.copy(),var_df_d13.copy(),ctrl_genes)
    RGR_sd=np.sqrt(RGR_var)
    #mf2_RGR,mf2_var=calculate_RGR(mean_df_d0_mf2.copy(),var_df_d0_mf2.copy(),mean_df_d13_mf2.copy(),var_df_d13_mf2.copy(),ctrl_genes)
    ### now combined fitness for mf1 and mf2
    # take mf1 and mf2 in one dtaframe
    ## calculate mean and variance

    viz_df=pd.DataFrame(index=RGR.index,columns=['HC22_RGR','HC22_var','HC22_sd','PG22_RGR','PG22_var','PG22_sd']);

    viz_df.loc[:,'HC22_RGR']=RGR.iloc[:,0]
    viz_df.loc[:,'HC22_var']=RGR_var.iloc[:,0]
    viz_df.loc[:,'HC22_sd']=RGR_sd.iloc[:,0]
    viz_df.loc[:,'PG22_RGR']=RGR.iloc[:,1]
    viz_df.loc[:,'PG22_var']=RGR_var.iloc[:,1]
    viz_df.loc[:,'PG22_sd']=RGR_sd.iloc[:,1]
    viz_df.loc[:,'PhenotypeHC22']='Not reduced'
    viz_df.loc[:,'PhenotypePG22']='Not reduced'
    ###
    viz_df.loc[:,'HC22_diff_max']=viz_df.loc[:,'HC22_RGR']+2*viz_df.loc[:,'HC22_sd']
    viz_df.loc[viz_df.loc[:,'HC22_diff_max']<-1,'PhenotypeHC22']='Reduced'
    viz_df.loc[:,'PG22_diff_max']=viz_df.loc[:,'PG22_RGR']+2*viz_df.loc[:,'PG22_sd']
    viz_df.loc[viz_df.loc[:,'PG22_diff_max']<-1,'PhenotypePG22']='Reduced'
    ###
    # ## get diffrence between HN and LN
    # viz_df.loc[:,'HC22minusPG22']=RGR.iloc[:,0]-RGR.iloc[:,1]
    # viz_df.loc[:,'HN_LN_var']=RGR_var.iloc[:,0]+RGR_var.iloc[:,1]
    #
    #
    #
    # pvals,fdrs=calPvalQval(viz_df.loc[:,'HNminusLN'],viz_df.loc[:,'HN_LN_var'])
    # viz_df['HNvsLN_pvalue']=pvals;
    # viz_df['HNvsLN_fdr']=fdrs[1];




    symbols=[]
    gene_names=[]
    for idx in RGR.index:
        tmp=db_df[db_df['Gene ID']==idx]
        if not tmp.empty:
            symbols.append(tmp['Gene Name or Symbol'].to_list()[0])
            gene_names.append(tmp['Product Description'].to_list()[0])
        else:
            symbols.append('NA')
            gene_names.append('NA')

    viz_df['Gene symbol']=symbols
    viz_df['Gene description']=gene_names

    if 'rgr_file' in plot_info.keys():
        print('writing rgr file---')
        #viz_df.to_csv(plot_info['rgr_file'],sep='\t')
        viz_df.to_excel(plot_info['rgr_file'],sheet_name='RGR_analysis')
    else:
        print('give file name of rgr file---')



def relative_growth_rate_analysis_arjun(df,manfest_df,prev_to_new,db_df,plot_info=None):
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
    grp_cols=['N','d','mf','b','t']
    day_pos=grp_cols.index('d')

    mean_df_d0,var_df_d0=propagate_error(rel_df,manfest_df,grp_cols,day_pos,days=['d0'])

    grp_cols=['N','d','mf','b','t']
    day_pos=grp_cols.index('d')
    mean_df_d13,var_df_d13=propagate_error(rel_df,manfest_df,grp_cols,day_pos,days=['d14'])


    if 'rel_file' in plot_info.keys():
        print('plotting propagated relative abundance')
        geneConv=plot_info['geneConv']
        plot_prop_rel_abun_nutrient(mean_df_d0,var_df_d0,mean_df_d13,var_df_d13,geneConv,plot_info)



    RGR,RGR_var=calculate_RGR_metabolic_pool(mean_df_d0.copy(),var_df_d0.copy(),mean_df_d13.copy(),var_df_d13.copy(),ctrl_genes)
    RGR_sd=np.sqrt(RGR_var)
    #mf2_RGR,mf2_var=calculate_RGR(mean_df_d0_mf2.copy(),var_df_d0_mf2.copy(),mean_df_d13_mf2.copy(),var_df_d13_mf2.copy(),ctrl_genes)
    ### now combined fitness for mf1 and mf2
    # take mf1 and mf2 in one dtaframe
    ## calculate mean and variance
    viz_df=pd.DataFrame(index=RGR.index,columns=['HN_RGR','HN_var','HN_sd','LN_RGR','LN_var','LN_sd']);

    viz_df.loc[:,'HN_RGR']=RGR.iloc[:,0]
    viz_df.loc[:,'HN_var']=RGR_var.iloc[:,0]
    viz_df.loc[:,'HN_sd']=RGR_sd.iloc[:,0]
    viz_df.loc[:,'LN_RGR']=RGR.iloc[:,1]
    viz_df.loc[:,'LN_var']=RGR_var.iloc[:,1]
    viz_df.loc[:,'LN_sd']=RGR_sd.iloc[:,1]

    ## get diffrence between HN and LN
    viz_df.loc[:,'HNminusLN']=RGR.iloc[:,0]-RGR.iloc[:,1]
    viz_df.loc[:,'HN_LN_var']=RGR_var.iloc[:,0]+RGR_var.iloc[:,1]



    pvals,fdrs=calPvalQval(viz_df.loc[:,'HNminusLN'],viz_df.loc[:,'HN_LN_var'])
    viz_df['HNvsLN_pvalue']=pvals;
    viz_df['HNvsLN_fdr']=fdrs[1];




    symbols=[]
    gene_names=[]
    for idx in RGR.index:
        tmp=db_df[db_df['Gene ID']==idx]
        if not tmp.empty:
            symbols.append(tmp['Gene Name or Symbol'].to_list()[0])
            gene_names.append(tmp['Product Description'].to_list()[0])
        else:
            symbols.append('NA')
            gene_names.append('NA')

    viz_df['Gene symbol']=symbols
    viz_df['Gene description']=gene_names

    if 'rgr_file' in plot_info.keys():
        print('writing rgr file---')
        #viz_df.to_csv(plot_info['rgr_file'],sep='\t')
        viz_df.to_excel(plot_info['rgr_file'],sheet_name='RGR_analysis')
    else:
        print('give file name of rgr file---')


    import pdb; pdb.set_trace()

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




def propagate_error(df,manfest_df,grp_cols,day_pos,days=['d0']):
    ''' We are going to calculate mean and SD for combined analyis from PCR to mosquitofeed '''
    df_log=np.log2(df)
    tmp_mn=pd.DataFrame(index=df.index)
    tmp_var=pd.DataFrame(index=df.index)
    tmp_mn_log=pd.DataFrame(index=df.index)
    ###
    final_mean=pd.DataFrame(index=df.index)
    final_var=pd.DataFrame(index=df.index)
    df_sample=pd.DataFrame(index=['num'])

    grp_cols.remove('mf')
    grp_cols.remove('b')
    for k,v in manfest_df.groupby(grp_cols).indices.items():
        if k[day_pos]in days:
            key='_'.join(k)
            tmp_mani=manfest_df.iloc[v].copy()

            for dr,dr_v in tmp_mani.groupby([ 'mf']).indices.items():
                dr='_'.join(dr)
                tmp_mn[key+'_'+dr]=df[tmp_mani.index[dr_v]].mean(axis=1).copy()
                tmp_var[key+'_'+dr]=df[tmp_mani.index[dr_v]].var(axis=1).copy()
            [mean_df,sd_max,var_max]=getCombined_mean_variance(tmp_mn,tmp_var,df_sample)

            # [mean_df,sd_max,var_max]=weighted_mean_variance(tmp_mn_log,tmp_var)
            final_mean[key]=mean_df.copy()
            final_var[key]=var_max.copy()

    final_mean_log=np.log2(final_mean)
    final_var_log=(final_var)/((final_mean*final_mean)*((np.log(10)**2)*np.log10(2)))
    return final_mean_log,final_var_log

def plot_pca(df,manfest_df):
    ''' arjun plot pca HN vs LN '''
    rel_df=df.copy()
    rel_df=rel_df.drop(columns=['Gene','Barcodes'])
    rel_df= rel_df + 1
    rel_df=rel_df.div(rel_df.sum(axis=0), axis=1)
    rel_df_log=np.log2(rel_df)
    N=['HN','LN']
    d=['d0','d14']


    dict_hn_ln=manfest_df.groupby(['N','d']).indices

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(rel_df.T.values)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [('HN','d0'), ('HN','d14'),('LN','d0'),('LN','d14')]
    colors = ['#dfc27d','#a6611a', '#80cdc1','#018571']

    for target, color in zip(targets,colors):
        indicesToKeep = dict_hn_ln[target]
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                   , principalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig('/Users/vpandey/projects/githubs/Fertility_screen/LN_HN_pca.pdf')
    plt.close()
    # import pdb; pdb.set_trace()

    # ### convert old pbanka to new ids
    # geneConv,old_to_new_ids,geneConv_new=getNewIdfromPrevID(rel_df.index,prev_to_new,db_df)
    # rel_df=rel_df.rename(old_to_new_ids, axis='index')
    # for k,v in manfest_df.groupby(grp_cols).indices.items():

def stepwiseAnalysis():
    ''' We are going do analyis of pool1 data of Claire '''

    ### these are the input files
    manifests_df=pd.read_excel(data_folder+"/HomingCas9_manifesto.xlsx")

    count_df=pd.read_csv(data_folder+ "/filtered_homing_count_table.csv",sep='\t')
    input_df=pd.read_excel(data_folder+'/HomingCas9_PilotSCreen_KO Vector list.xlsx')

    #### end of the input section
    # final_count_df: read1 and read2 are added
    # final_count_df_two_read: reads are sperated
    # manfest_df: maifest_df
    final_count_df,final_count_df_des,final_count_df_two_read,manfest_df=preprocessing(manifests_df,count_df)

    ####  Dropouts and input check
    # input_df: these are genes which was used for pool phenotypes
    percent=0.95 ## this parameters is used to test whether count is too small for 90 % of input samples. Those will be deleted.

    filtered_count_df,filtered_df_read,filtered_count_df_des=filter_input_dropout(final_count_df,final_count_df_des,final_count_df_two_read,input_df,manfest_df,percent)

    ######  write filtered and unfiltered files
    # final_count_df_two_read.to_csv(out_folder+"/unfilterd_count_matrix_pool2.txt",sep='\t')

    filtered_count_df_des.to_csv(out_folder+"/filterd_count_homing_cas9.txt",sep='\t')

    # plot_pca(filtered_count_df,manfest_df)

    ### we are going to perform relative abundance analysis
    ## prev_to_new this is the pickle information which is used when we change old to new ID
    ## db_df: this is the dataframe contains name and description

    geneConv,old_to_new_ids,geneConv_new=getNewIdfromPrevID(filtered_count_df.index,prev_to_new,db_df)

    plot_info={'pool':'homing cas9','rel_file':out_folder+'/homing_cas9_relative_abundance.pdf','d':['d0','d14'],
    'mf':['mf1','mf2'],'N':['HC22','PG22'],'geneConv':geneConv_new,
    'control_genes':['PBANKA_0308200','PBANKA_1315100'],'rgr_file':out_folder+'/cas9_pool_RGR.xlsx'}
    relative_growth_rate_analysis_homing(filtered_count_df,manfest_df,prev_to_new,db_df,plot_info)
    #PBANKA_0616700





    ### we are going to start analysing
    # ### now we are going to combine two dataframe with categorical data sets
    # cmd_df=final_df.T.join(manifests_df)




if __name__ == '__main__':
    stepwiseAnalysis()
