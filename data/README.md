# Download Datasets

The `data` folder contains DLPFC, Liver, MOB and PDAC.

## DLPFC
```bash
wget https://zenodo.org/records/11114959/files/DLPFC.zip
unzip DLPFC.zip
```
## HLN
download ST data from https://drive.google.com/drive/folders/1ns-EsWBu-SNrJ39j-q-AFIV5U-aXFwXf.
```bash
wget https://cell2location.cog.sanger.ac.uk/paper/integrated_lymphoid_organ_scrna/RegressionNBV4Torch_57covariates_73260cells_10237genes/sc.h5ad
```
```python
adata_sc = sc.read('sc.h5ad')
adata_sc.var['SYMBOL'] = adata_sc.var.index
# rename 'GeneID-2' as necessary for your data
adata_sc.var.set_index('GeneID-2', drop=True, inplace=True)
adata_sc.write('sc.h5ad', compression='gzip')
```
## Liver
```bash
wget https://zenodo.org/records/11114959/files/Liver.zip
unzip Liver.zip
```
## MOB
```bash
wget https://zenodo.org/records/11114959/files/MOB.zip
unzip MOB.zip
```
## PDAC
```bash
wget https://zenodo.org/records/11114959/files/PDAC.zip
unzip PDAC.zip
```