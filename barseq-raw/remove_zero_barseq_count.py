import sys
import pandas as pd
def remove_unwanted_count(csv_file,out_file):

    # df=pd.read_csv("/Users/vikash/P13459/results_P13459/Experiment/barcode_counts_table.csv")
    df=pd.read_csv(csv_file)
    df.drop(df[df['Barcodes']=="_other"].index, inplace = True)

    # df2=df.drop(columns=['Unnamed: 0'])
    # arr=df2.to_numpy()
    # pheno=pd.read_csv("/Users/vikash/Documents/Projects/MalariaCellAtlas-master/Expression_Matrices/Smartseq2/SS2_pheno.csv",encoding='latin1')
    # pheno_df=pheno.copy()
    # pheno.set_index('sample_id',inplace=True)
    #
    # values={'EEF':'Liver_stage','male':'Male_Gametocyte','female':'Female_Gametocyte','oocyst':'Oocyst','ook':'Ookinete', 'ookoo':'Ookinete','sgSpz':'Gland_sporozite','bbSpz':'Injected_sporozite'}
    # values2={'Liver_stage':0,'Merozoite':1, 'Ring': 2, 'trophozoite':3, 'schizont':4, 'Male_Gametocyte':5, 'Female_Gametocyte':6, 'Ookinete':7, 'Oocyst':8, 'Gland_sporozite':9, 'Injected_sporozite':10,'NA':11, 'outlier':12,'Ookinete2':13}
    #
    # df1=pheno.loc[df.columns[1:],['specstage']]
    # df1=df1.fillna('NA')
    # df1=df1.replace(values)
    df1=df.loc[:, (df != 0).any(axis=0)] # drop all zero columns
    # df2=df1.set_index([ 'Gene',     'Barcodes'])
    df2=df1.set_index([ 'Unnamed: 0',     'Barcodes'])
    df3 = df2[(df2.T != 0).any()]
    df3.to_csv(out_file,sep='\t')




if __name__ == '__main__':
    csv_file=sys.argv[1] # barcode with zero values
    out_file=sys.argv[2] # outfile with removed zeros
    remove_unwanted_count(csv_file,out_file)
