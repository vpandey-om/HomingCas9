import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.cm as cm
from pylab import *
from plotnine import *
# plt.style.use('seaborn-whitegrid')
cmap = cm.get_cmap('seismic', 5)
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
## import final xlsx file then plot s curve
# cas9_df=pd.read_excel('/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/cas9_pool_RGR_shortname.xlsx')
cas9_df=pd.read_excel('/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/cas9_pool_RGR2.xlsx')



# cas9_df['Gene ID']=cas9_df['pbanka_id'].copy()


#colorDict={'Not reduced':'#66C2A5','Reduced':'#FC8D62'}
colorDict={'pg':'#66C2A5','hc':'#FC8D62'}
shapeDict={'pg':'o','hc':'o'}
fillDict={'pg':'#66C2A5','hc':'#ffffff00'}
fillDict1={'S':'#ffffff00'}
sizeDict={'pg':2,'hc':2.5}
changeDict={'NS':'#66C2A5','S':'#FC8D62'}

## choice of color
order_df=pd.read_excel('/Users/vpandey/projects/gitlabs/arjunproject/homing/data/Genes categorized on phenotype2.xlsx')

sorter=order_df['Gene ID'].copy()
sorterIndex = dict(zip(sorter, range(len(sorter))))

sorter=order_df['Short name '].copy()
sorterIndex2 = dict(zip(range(len(sorter)),sorter))

cas9_df=pd.concat([cas9_df,order_df],axis=1)

cas9_df['Gene ID'] = cas9_df['pbanka_id'].map(sorterIndex)

cas9_df['Gene ID2'] = cas9_df['Gene ID'].copy()
cas9_df['Short name ']=cas9_df['Gene ID2'].map(sorterIndex2)

# get male dataframe

### test off target


def plot_offtarget2(df,plot_df):
    df.loc[df['pval']<1e-8,'pval']=1e-8
    df['-log10(pval)']=-np.log10(df['pval'])
    # remove_items=['APC3','PBANKA_0312700','MAPK2']
    # df = df[~df['Short name'].isin(remove_items)]
    # ### remove some values 
    # save_list=['ATPbeta','AP2-O4 ','PBANKA_0916000','GR','SOAP','PBANKA_1006400']
    # df['label']=df['Short name']
    # df['color']='S'
    # for ind in df.index:
    #     if df.loc[ind,'label'] not in save_list:
    #         df.loc[ind,'label']=''
    #         df.loc[ind,'color']='NS'
    # p=sns.scatterplot(data=df, x="hc", y="pg", hue="log2FC", palette="viridis",legend = False)
    # # Add color scale bar using Matplotlib
    # sm = plt.cm.ScalarMappable(cmap='viridis')
    # sm.set_array(df["log2FC"].values)
    # cbar = plt.colorbar(sm)

 
    p=( ggplot(aes(x='hc',y='pg',color='log2FC'),df)
        # + geom_errorbar( width = 0.2)
        + geom_point(size=2.5)
        # + geom_text(aes(label='label',size=8,x='FC-0.25',y='-log10(pval)-0.25'))
        # + scale_color_gradient2(low='blue', mid='white', high='red')
        # # + labs(title='HC22 conversion rate')
        # + scale_color_manual(changeDict)
        # + scale_shape_manual(shapeDict)
        # + scale_fill_manual(fillDict)
        # + scale_size_manual(sizeDict)
        # + scale_x_discrete(limits=labels)
        # + scale_color_gradient(low='blue', high='red')
        + scale_color_gradient2(low="blue", mid="white",
                     high="red", space ="Lab" )
        + theme(
        axis_text_x=element_text(size=8,colour="black",family='Arial'),
        axis_text_y=element_text(size=8,colour="black",family='Arial'),
        axis_line=element_line(size=1, colour="black"),
        figure_size=(3.22*1.30, 2.4*1.37),
        # panel_grid_major=element_line(colour="#d3d3d3"),
        # panel_grid_minor=element_blank(),
        # legend_position='none',
        panel_border=element_blank(),
        panel_background=element_blank())
        # + xlab('log2FC')
        # + ylab('Oocyst conversion rate',size=11,family='Arial')
        # + labs(color='Vector')
        # + geom_text(aes(x =xlab, y='-log10(pval)', label = 'compounds'), color = '#252525', size=4
        # ,position=position_jitter(width=0.05,height=0.05))
        )
   
    p.save(plot_df)
    


def plot_offtarget(df,plot_df,vmin,vmax):

    df.loc[df['pval']<1e-8,'pval']=1e-8
    df['-log10(pval)']=-np.log10(df['pval'])

    # remove_items=['APC3','PBANKA_0312700','MAPK2']
    # df = df[~df['Short name'].isin(remove_items)]
    # ### remove some values 
    # save_list=['ATPbeta','AP2-O4 ','PBANKA_0916000','GR','SOAP','PBANKA_1006400']
    # df['label']=df['Short name']
    # df['color']='S'
    # for ind in df.index:
    #     if df.loc[ind,'label'] not in save_list:
    #         df.loc[ind,'label']=''
    #         df.loc[ind,'color']='NS'
    import math
    # hue_order = df.sort_values('log2FC')['log2FC'].unique()
    sm = cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # Get the hexadecimal color representation for each value
    cls1= sm.to_rgba(df.sort_values('log2FC')['log2FC'].values)
    df['color']= cls1.tolist()
    tmp=df[df['pbanka']=='PBANKA_093290']
    p=sns.scatterplot(data=df, x="hc", y="pg", hue="log2FC",palette=df['color'].to_list(),legend = False,linewidth=0,alpha = 0.7)
    # Add color scale bar using Matplotlib
    # Add a text label to a specific point
    # plt.text(tmp['hc'].to_list()[0], tmp['pg'].to_list()[0], 'G3PDH', ha='center', va='center',alpha=0.5)
    plt.annotate('G3PDH', (tmp['hc'].to_list()[0], tmp['pg'].to_list()[0]), xytext=(tmp['hc'].to_list()[0]+2,tmp['pg'].to_list()[0]+2),
             textcoords='offset points', arrowprops=dict(arrowstyle='->'))

    # sm = plt.cm.ScalarMappable(cmap='viridis')
    # # # # sm.set_array()
    # cbar = plt.colorbar(sm)
    # Get the current axes
    # Get the ScalarMappable object for the color scale
    # sm = plt.cm.ScalarMappable(cmap='viridis')
   
    # # cbar = plt.colorbar(sm)

    # # # Create a normalize object to map values to the range [0, 1]
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    # plt.colorbar(sm)

    # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # # # # Create a ScalarMappable object for the color scale
    # sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)

    # # # Set the colorbar values and label
    # sm.set_array([])
    plt.colorbar(sm, label='log2FC')
    

    # plt.axis('equal')
    maxi=math.ceil(df[['hc','pg']].max().max())
    mini=math.floor(df[['hc','pg']].min().min())
    plt.ylim(mini,maxi)
    plt.xlim(mini,maxi)


    # p=( ggplot(aes(x='hc',y='pg',color='log2FC'),df)
    #     # + geom_errorbar( width = 0.2)
    #     + geom_point(size=2.5)
    #     # + geom_text(aes(label='label',size=8,x='FC-0.25',y='-log10(pval)-0.25'))
    #     + scale_color_gradient2(low='blue', mid='white', high='red')
    #     # # + labs(title='HC22 conversion rate')
    #     # + scale_color_manual(changeDict)
    #     # + scale_shape_manual(shapeDict)
    #     # + scale_fill_manual(fillDict)
    #     # + scale_size_manual(sizeDict)
    #     # + scale_x_discrete(limits=labels)
    #     # + scale_color_gradient(low='blue', high='red')
    #     + theme(
    #     axis_text_x=element_text(size=8,colour="black",family='Arial'),
    #     axis_text_y=element_text(size=8,colour="black",family='Arial'),
    #     axis_line=element_line(size=1, colour="black"),
    #     figure_size=(3.22*1.30, 2.4*1.37),
    #     # panel_grid_major=element_line(colour="#d3d3d3"),
    #     # panel_grid_minor=element_blank(),
    #     # legend_position='none',
    #     panel_border=element_blank(),
    #     panel_background=element_blank())
    #     # + xlab('log2FC')
    #     # + ylab('Oocyst conversion rate',size=11,family='Arial')
    #     # + labs(color='Vector')
    #     # + geom_text(aes(x =xlab, y='-log10(pval)', label = 'compounds'), color = '#252525', size=4
    #     # ,position=position_jitter(width=0.05,height=0.05))
    #     )
   
    # p.save(plot_df)
    # Get the current axes
    ax = plt.gca()

    # Set the font family to Arial
    font_path = fm.findfont(fm.FontProperties(family='Arial'))
    font_prop = fm.FontProperties(fname=font_path, size=12)
    ax.set_xlabel('+homing', fontproperties=font_prop)
    ax.set_ylabel('-homing', fontproperties=font_prop)

    plt.savefig(plot_df)
    plt.close()
def offTargetAnalysis(offdf):
    shortName=[]
    hclist=[]
    pglist=[]
    pvals=[]
    FC=[]

    for item in offdf.index:
        hc=offdf.loc[item,'HC22_d0_NA_mean']
        hclist.append(hc)
        pg=offdf.loc[item,'PG22_d0_NA_mean']
        pglist.append(pg)
        var_hc=offdf.loc[item,'HC22_d0_NA_var']
        var_pg=offdf.loc[item,'PG22_d0_NA_var']
        
        shortName.append(offdf.loc[item,'pbanka_id'])
        #### claculate fold change
        ## claculate pvalues 
        # Calculate the z-score
        z_score = (hc - pg) / ((var_hc + var_pg)**0.5)
        FC.append(hc-pg)
        # Calculate the p-value using two-sample t-test
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        pvals.append(p_value)


    res=pd.DataFrame()
    res['pbanka']=shortName
    res['hc']=hclist
    res['pg']=pglist
    res['pval']=pvals
    res['log2FC']=FC

    return res


def offTargetAnalysisCuvett(offdf):
    shortName=[]
    hclist=[]
    pglist=[]
    pvals=[]
    FC=[]
    for item in offdf.index:
        hc=offdf.loc[item,'t1_mean']
        hclist.append(hc)
        pg=offdf.loc[item,'t2_mean']
        pglist.append(pg)
        var_hc=offdf.loc[item,'t1_var']
        var_pg=offdf.loc[item,'t2_var']
        
        shortName.append(offdf.loc[item,'pbanka_id'])
        #### claculate fold change
        ## claculate pvalues 
        # Calculate the z-score
        z_score = (hc - pg) / ((var_hc + var_pg)**0.5)
        FC.append(hc-pg)
        # Calculate the p-value using two-sample t-test
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        pvals.append(p_value)


    res=pd.DataFrame()
    res['pbanka']=shortName
    res['hc']=hclist
    res['pg']=pglist
    res['pval']=pvals
    res['log2FC']=FC

    return res


def plotScatter(df,x,y,type,xlab1,ylab1,file):
    colorDict1={'PG':'#66C2A5','HC':'#FC8D62'}

    p=( ggplot(aes(x=x, y=y,color=type),df)
        + geom_point()
        # # + labs(title='HC22 conversion rate')
        + scale_color_manual(colorDict1)
        #+ scale_x_discrete(limits=labels)
        + theme(axis_text_x=element_text(rotation=45, hjust=1),
        axis_line=element_line(size=1, colour="black"),
        # panel_grid_major=element_line(colour="#d3d3d3"),
        # panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank())
        + xlab(xlab1)
        + ylab(ylab1)
        + labs(color='Types')
        + xlim(-13, 0)
        + ylim(-6.2, -3)
        # + geom_text(aes(x =xlab, y='-log10(pval)', label = 'compounds'), color = '#252525', size=4
        # ,position=position_jitter(width=0.05,height=0.05))
        )
    p.save(file)

offtarget_df=pd.read_excel('/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/offtargetResults.xlsx')
file='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/offtarget.pdf'
#offtarget_df['color']='#636363'
offdf=offTargetAnalysis(offtarget_df)
vmin=offdf['log2FC'].min()
vmax=offdf['log2FC'].max()
offdf_cuv=offTargetAnalysisCuvett(offtarget_df)
# Normalize values to range [0, 1]
# offdf['log2FC'] = (offdf['log2FC'].values - vmin) / (vmax - vmin)
# offdf_cuv['log2FC'] = (offdf_cuv['log2FC'].values - vmin) / (vmax - vmin)

# Create a ScalarMappable object for the colormap and range
sm = cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# Get the hexadecimal color representation for each value
cls1= sm.to_rgba(offdf['log2FC'].values)
offdf['color']= cls1.tolist()
# # Create a figure and axis
# fig, ax = plt.subplots()

# # Plot a scatter plot with RGBA colors
# ax.scatter([i+1 for i in range(len(colors1))], [i+1 for i in range(len(colors1))], c=colors1.tolist())

# # Show the plot
# plt.show()

cls1= sm.to_rgba(offdf_cuv['log2FC'].values)
offdf_cuv['color']=cls1.tolist()
plot_off='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/off_target_input.pdf'
plot_offtarget(offdf,plot_off,vmin,vmax)

plot_off='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/off_target_cuvett.pdf'
plot_offtarget(offdf_cuv,plot_off,vmin,vmax)

res_df=pd.DataFrame()
res_df['d0']=offtarget_df['HC22_d0_NA_mean'].to_list()+offtarget_df['PG22_d0_NA_mean'].to_list()
res_df['t']=offtarget_df['t1_mean'].to_list()+offtarget_df['t2_mean'].to_list()
res_df['type']=['HC']*offtarget_df.shape[0]+['PG']*offtarget_df.shape[0]

plotScatter(res_df,'d0','t','type','Input (day0)','Cuvettes',file)
# file='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/pg22_offtarget.pdf'
# plotScatter(offtarget_df,'PG22_d0_NA_mean','t2_mean','Input (day0)','Cuvettes',file)


##




def plotScurveCombine(df,plot_df='test.pdf'):

    # sorted_df=df.sort_values(by=['Gene ID']).copy()
    # sorted_df=df.copy()
    #
    # sorted_df['x']=np.arange(sorted_df.shape[0])+1
    # labels=sorted_df['Short name '].to_list()
    manufacturer_cat = pd.Categorical(df['Short name '], categories=order_df['Short name '])
    df = df.assign(orername= manufacturer_cat)
    ordertype = pd.Categorical(df['type'], categories=['pg','hc'])
    df = df.assign(ordertype= ordertype)
    p=( ggplot(aes(x='orername', y='rgr',ymin = 'min', ymax = 'max',group='ordertype',color='ordertype'),df)
        + geom_errorbar( width = 0.2,position=position_dodge(0.5))
        + geom_point(position=position_dodge(0.5))
        + geom_line(y=-2,linetype = "dashed",color='red')
        # + labs(title='HC22 conversion rate')
        + scale_color_manual(colorDict)

        # + scale_x_discrete(limits=labels)
        + theme(axis_text_x=element_text(rotation=45, hjust=1),
        axis_line=element_line(size=1, colour="black"),
        # panel_grid_major=element_line(colour="#d3d3d3"),
        # panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank())   
        + xlab('Genes')
        + ylab('Oocyst conversion rate')
        + labs(color='Vector')
        # + geom_text(aes(x =xlab, y='-log10(pval)', label = 'compounds'), color = '#252525', size=4
        # ,position=position_jitter(width=0.05,height=0.05))
        )

    p.save(plot_df)

    


def plotScurveCombine2(df,plot_df='test.pdf'):

    # sorted_df=df.sort_values(by=['Gene ID']).copy()
    # sorted_df=df.copy()
    #
    # sorted_df['x']=np.arange(sorted_df.shape[0])+1
    # labels=sorted_df['Short name '].to_list()
    manufacturer_cat = pd.Categorical(df['Short name '], categories=order_df['Short name '])
    df = df.assign(orername= manufacturer_cat)
    ordertype = pd.Categorical(df['type'], categories=['pg','hc'])
    df = df.assign(ordertype= ordertype)
    p=( ggplot(aes(x='orername', y='rgr',ymin = 'min', ymax = 'max',shape='ordertype',color='ordertype',fill='ordertype',size='ordertype'),df)
        # + geom_errorbar( width = 0.2)
        + geom_point()
        + geom_line(y=-2,linetype = "dashed",color='red')
        # + labs(title='HC22 conversion rate')
        + scale_color_manual(colorDict)
        + scale_shape_manual(shapeDict)
        + scale_fill_manual(fillDict)
        + scale_size_manual(sizeDict)
        # + scale_x_discrete(limits=labels)
        + theme(axis_text_x=element_text(rotation=45, hjust=1,size=8,colour="black",family='Arial'),
        axis_line=element_line(size=1, colour="black"),
        figure_size=(3.22*1.30, 2.4*1.37),
        # panel_grid_major=element_line(colour="#d3d3d3"),
        # panel_grid_minor=element_blank(),
        legend_position='none',
        panel_border=element_blank(),
        panel_background=element_blank())
        # + xlab('Genes')
        # + ylab('Oocyst conversion rate',size=11,family='Arial')
        # + labs(color='Vector')
        # + geom_text(aes(x =xlab, y='-log10(pval)', label = 'compounds'), color = '#252525', size=4
        # ,position=position_jitter(width=0.05,height=0.05))
        )
   
    p.save(plot_df)


def plotScurve(df,plot_df='test.pdf'):
    ## sort dataframe
    df['min']=df['rgr']+df['sd']
    df['max']=df['rgr']-df['sd']
    sorted_df=df.sort_values(by=['Gene ID']).copy()
    # sorted_df=df.copy()

    sorted_df['x']=np.arange(sorted_df.shape[0])+1
    labels=sorted_df['Short name '].to_list()
    p=( ggplot(aes(x='x', y='rgr',ymin = 'min', ymax = 'max',color='pheno'),sorted_df)
        + geom_errorbar( width = 0.2)
        + geom_point()
        + geom_line(y=-1,linetype = "dashed",color='red')
        # + labs(title='HC22 conversion rate')
        + scale_color_manual(colorDict)
        + scale_x_discrete(limits=labels)
        + theme(axis_text_x=element_text(rotation=90, hjust=1),
        axis_line=element_line(size=1, colour="black"),
        # panel_grid_major=element_line(colour="#d3d3d3"),
        # panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank())
        + xlab('Genes')
        + ylab('Oocyst conversion rate')
        + labs(color='Phenotype')
        # + geom_text(aes(x =xlab, y='-log10(pval)', label = 'compounds'), color = '#252525', size=4
        # ,position=position_jitter(width=0.05,height=0.05))
        )

    p.save(plot_df)
    # # get for each phenoype and plot
    # ax = subplot(1,1,1)
    #
    # for i,item in enumerate(pheno_names):
    #     p_df=sorted_df[sorted_df['pheno']==item].copy()
    #     h1=ax.errorbar(p_df['x'].values, p_df['rgr'].values, yerr=p_df['sd'].values*2, fmt='o', color=pheno_colors[i],
    #              ecolor=pheno_colors[i], elinewidth=1, capsize=3,label=pheno_legends[i],ms=2)
    #     plt.xticks(p_df['x'].values, p_df['Gene symbol'].values)
    #     plt.xticks(rotation = 45)
    #
    # # for i,item in enumerate(pheno_names):
    # #     p_df=sorted_df[sorted_df['pheno']==item].copy()
    # #     h1=ax.errorbar(p_df['x'].values, p_df['rgr'].values, yerr=p_df['sd'].values*2, fmt='o', color=pheno_colors[i],
    # #              ecolor=pheno_colors[i], elinewidth=1, capsize=3,label=pheno_legends[i],ms=2)
    # #     plt.xticks(p_df['x'].values, p_df['Gene symbol'].values)
    # #     plt.xticks(rotation = 45)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1])
    # plt.grid(axis='y', color='0.95')
    # plt.xlabel('Ranked genes [1-%d]'%sorted_df.shape[0], fontsize=16)
    # plt.ylabel('Conversion rate', fontsize=16)
    # plt.ylim(-18, 8)
    # # plt.xlim(-14, 1150)
    # ## reoder legends
    #
    # plt.savefig(plot_df, dpi=300)
    # plt.close()


def plot_half_circle(df,plot_df='test.df'):
    ''' we would like to fill half circles  '''
    ## get male female redcued
    fig, ax = plt.subplots(figsize=[6.4, 6.4])
    df['label']=df['Gene symbol']
    df.loc[df['PhenotypeHC22']=='Not reduced','label']=''
    #"#f7f7f7"
    # ax.scatter(df['g145480_RGR'], df['GCKO2_RGR'], c=df['male_color'], s=30,edgecolor='#F5F5F5', alpha=0.6,linewidth=0.2,marker=MarkerStyle("o", fillstyle="right"))
    # ax.scatter(df['g145480_RGR'], df['GCKO2_RGR'], c=df['female_color'], s=30,edgecolor='#F5F5F5', alpha=0.6,linewidth=0.2,marker=MarkerStyle("o", fillstyle="left"))
    #
    for i in df.index:
        marker_style = dict(color=df.loc[i,'pg22_color'],markersize=6, markerfacecoloralt=df.loc[i,'hc22_color'],markeredgecolor='#F5F5F5',alpha=0.8,markeredgewidth=0.2)
        ax.plot(df.loc[i,'HC22_RGR'],  df.loc[i,'PG22_RGR'], 'o',fillstyle='left', **marker_style)
        ax.text(df.loc[i,'HC22_RGR'],  df.loc[i,'PG22_RGR'], df.loc[i,'label'])
    plt.xlabel('HC Oocyst conversion Rate',fontsize=15)
    plt.ylabel('PG Oocyst conversion Rate',fontsize=15)
    # ax.set_xlabel('Male fertility',fontsize=15)
    # ax.set_ylabel('Female fertility',fontsize=15)
    # ax.tick_params(axis='y',labelsize=15)
    # ax.tick_params(axis='x',labelsize=15)
    # ax.set_xlim(-13,3)
    # ax.set_ylim(-13,3)
    plt.xlim(-13,3)
    plt.ylim(-13,3)
    plt.savefig(plot_df, dpi=300)
    plt.close()
    #plt.show()

def calculatePvalFC(odf,pgdf,hcdf):
    ''' calculate pvalue and logFC'''
    shortName=[]
    FC=[]
    pvals=[]
    for item in odf['Short name '].to_list():
        tmp1=pgdf[pgdf['Short name ']==item]
        tmp2=hcdf[hcdf['Short name ']==item]
        rgr_pg=tmp1['rgr'].to_list()[0]
        sd_pg=tmp1['sd'].to_list()[0]
        rgr_hc=tmp2['rgr'].to_list()[0]
        sd_hc=tmp2['sd'].to_list()[0]
        shortName.append(item)
        #### claculate fold change
        FC.append(rgr_hc-rgr_pg)
        ## claculate pvalues 
        # Calculate the z-score
        z_score = (rgr_hc - rgr_pg) / ((sd_pg**2 + sd_hc**2)**0.5)
        # Calculate the p-value using two-sample t-test
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        pvals.append(p_value)

    res=pd.DataFrame()
    res['Short name']=shortName
    res['FC']=FC
    res['pval']=pvals
    return res


def plot_volcano(df,plot_df):
    manufacturer_cat = pd.Categorical(df['Short name'], categories=df['Short name'])
    df = df.assign(orername= manufacturer_cat)
    df.loc[df['pval']<1e-8,'pval']=1e-8
    df['-log10(pval)']=-np.log10(df['pval'])
    remove_items=['APC3','PBANKA_0312700','MAPK2']
    df = df[~df['Short name'].isin(remove_items)]
    ### remove some values 
    save_list=['ATPbeta','AP2-O4 ','PBANKA_0916000','GR','SOAP','PBANKA_1006400']
    df['label']=df['Short name']
    df['color']='S'
    df['fill']='S'
    for ind in df.index:
        if df.loc[ind,'label'] not in save_list:
            df.loc[ind,'label']=''
            df.loc[ind,'color']='NS'

    
    p=( ggplot(aes(x='FC', y='-log10(pval)',fill='fill'),df)
        # + geom_errorbar( width = 0.2)
        + geom_point(size=2.5)
        # + geom_text(aes(label='label',size=8,x='FC-0.25',y='-log10(pval)-0.25'))
        + geom_line(y=-np.log10(0.05),linetype = "dashed",color='black')
        # # + labs(title='HC22 conversion rate')
        # + scale_color_manual(changeDict)
        # + scale_shape_manual(shapeDict)
        + scale_fill_manual(fillDict1)
        # + scale_size_manual(sizeDict)
        # + scale_x_discrete(limits=labels)
        + theme(
        axis_text_x=element_text(size=8,colour="black",family='Arial'),
        axis_text_y=element_text(size=8,colour="black",family='Arial'),
        axis_line=element_line(size=1, colour="black"),
        figure_size=(3.22*1.30, 2.4*1.37),
        # panel_grid_major=element_line(colour="#d3d3d3"),
        # panel_grid_minor=element_blank(),
        legend_position='none',
        panel_border=element_blank(),
        panel_background=element_blank())
        + xlab('log2FC')
        # + ylab('Oocyst conversion rate',size=11,family='Arial')
        # + labs(color='Vector')
        # + geom_text(aes(x =xlab, y='-log10(pval)', label = 'compounds'), color = '#252525', size=4
        # ,position=position_jitter(width=0.05,height=0.05))
        )
   
    p.save(plot_df)




## preprocessing

hc22_cols=['HC22_RGR','HC22_sd','PhenotypeHC22','Short name ','Gene ID']

hc22_df=cas9_df[hc22_cols].copy()
hc22_df=hc22_df.rename(columns = {'HC22_RGR': 'rgr','HC22_sd':'sd','PhenotypeHC22':'pheno'})




#plotScurve(male_df) ## bydefault

# #pheno_names=['Reduced','Not Reduced','No Power']
# # pheno_names=['No Power','Not Reduced','Reduced',]
# # pheno_colors=['#cccccc','#b2e2e2','#d8b365',]
# # pheno_legends=['No power','Not reduced','Reduced']
# pheno_labels=['Reduced','Not reduced','No power']

# pheno_names=['Reduced','Not reduced']
# pheno_colors=['#FC8D62','#66C2A5']
# pheno_legends=['Reduced','Not reduced']

# pheno_names=['Not reduced','Reduced',]
# pheno_colors=['#66C2A5','#FC8D62']
# pheno_legends=['Not redueced','Reduced']

plot_df='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/hc22_S2.pdf'

# plotScurve(hc22_df,plot_df)

pg22_cols=['PG22_RGR','PG22_sd','PhenotypePG22','Short name ','Gene ID']
pg22_df=cas9_df[pg22_cols].copy()
pg22_df=pg22_df.rename(columns = {'PG22_RGR': 'rgr','PG22_sd':'sd','PhenotypePG22':'pheno'})
plot_df='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/pg22_S2.pdf'

# plotScurve(pg22_df,plot_df)
pg22_df['max']=pg22_df['rgr']+pg22_df['sd']
pg22_df['min']=pg22_df['rgr']-pg22_df['sd']
hc22_df['max']=hc22_df['rgr']+hc22_df['sd']
hc22_df['min']=hc22_df['rgr']-hc22_df['sd']
hc22_df['type']='hc'
pg22_df['type']='pg'
cmd_df=pd.concat([hc22_df,pg22_df],keys=['Short name ','Short name '])

plot_df='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/combine2.pdf'
volcano='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/volcano.pdf'
pvaldf=calculatePvalFC(order_df,pg22_df,hc22_df)
pvaldf.to_excel('/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/FC_data.xlsx')
plot_volcano(pvaldf,volcano)
import pdb;pdb.set_trace()
plotScurveCombine2(cmd_df,plot_df)
import pdb;pdb.set_trace()


## get common between male and female

### plot half circle
## colors
common_df=cas9_df.copy()
common_df['hc22_color']=common_df['PhenotypeHC22'] ## Not reduced
common_df['pg22_color']=common_df['PhenotypePG22']  ## Not reduced

common_df=common_df.replace({'hc22_color': colorDict,
'pg22_color': colorDict})
plot_df='/Users/vpandey/projects/gitlabs/arjunproject/homing/Figures/half_circle_anot.pdf'
plot_half_circle(common_df,plot_df)





##### we

# def plotMaleFemaleScatter(df):
#     # fig = plt.figure(figsize=(8,8))
#     fig, ax = plt.subplots(figsize=(9, 9))
#
#     ###
#     # change df_values
#
#     df['145480_d13_pheno']=df['145480_d13_pheno'].replace({'E': 'IM', 'NE': 'FM','NA':'RM'})
#     df['GCKO2_d13_pheno']=df['GCKO2_d13_pheno'].replace({'E': 'IF', 'NE': 'FF', 'NA': 'RF'})
#     # viz_df['Published_cross_phenotype']=viz_df['Published_cross_phenotype'].replace({'N': 'NA'})
#
#     # cmap={'FM':"#66c2a5", 'RM':"#8da0cb", 'IM':"#fc8d62"}
#     cmap_male={'FM':"#1b9e77", 'RM':"#7570b3", 'IM':"#d95f02"}
#     cmap_female={'FF':"#1b9e77", 'RF':"#7570b3", 'IF':"#d95f02"}
#
#
#     for i,item in enumerate(df['GCKO2_d13_pheno'].to_list()):
#         marker_style = dict(color=cmap_female[item],markersize=6, markerfacecoloralt=cmap_male[df['145480_d13_pheno'][i]],markeredgecolor='white',alpha=0.8)
#         ax.plot(df['GCKO2_d13_rel'][i],df['145480_d13_rel'][i], 'o',fillstyle='left', **marker_style)
#
#
#     ax.set_xlabel('Relative growth rate (Female)',fontsize=15)
#     ax.set_ylabel('Relative growth rate (Male)',fontsize=15)
#     ax.tick_params(axis='y',labelsize=12)
#     ax.tick_params(axis='x',labelsize=12)
#     ax.set_xlim(-11,3)
#     ax.set_ylim(-11,3)
#     # handles, labels = ax.get_legend_handles_labels()
#     # ax.legend(handles, labels)
#     infertile= mpatches.Patch(color="#fc8d62", label='Infertility (F/M)')
#     fertile=mpatches.Patch(color="#66c2a5", label='Normal fertility (F/M)')
#     reduced_fertile=mpatches.Patch(color="#8da0cb", label='Reduced fertility (F/M)')
#     plt.legend(handles=[infertile,fertile,reduced_fertile],loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3, borderaxespad=0, frameon=False, prop={'size': 6})
#
#     fig.savefig(plot_folder + "scatter_plot_male_female_RGR_pool1.pdf")
#
#

####
