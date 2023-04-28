import sys # Remover futuramente
import logging # Remover futuramente
import pandas as pd
import numpy as np
#from tensorflow import keras
import keras
from tensorflow_core.python.keras.models import Sequential, model_from_json, load_model # tensorflow.python.
from tensorflow_core.python.keras.layers import Dense, BatchNormalization,Dropout #tensorflow.python.
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path # Remover futuramente
import datetime
from sklearn.externals import joblib

# VARIABLES
#path = Path(__file__).parent

def model_init(model_file):#model_weights_file, model_arquitecture_file)
    # Carregar modelo
    loaded_model = load_model(model_file)
    return loaded_model

def model_get_input_from_json2(data):
    #df = pd.DataFrame.from_dict(data)
    #print(data)

    chave = data['cChaveAcessoNFCab']
    qtde_itens = data['nQtdItensNFCab']
    centro = data['cCentroCab']
    emissao = data['dEmissaoNFCab']
    vencimento = data['dVencimentoDuplicataCab']
    fornecedor = data['cFornecedorCab']
    cnpj = data['cCNPJNFCab']
    tipo_pc = data['cTipoPedidoCab']
    incoterms = data['cIncotermsCab']
    peso = data['nPesoTotalNFCab']
    valor = data['nVlrTotalBrutoNFCab']
    mod_frete = data['cModalidadeFreteCab']
    frente = data['tFrenteCab']
    cfop = data['cCFOPItem']

    #chave = [item['cChaveAcessoNFCab'] for item in data]
    #qtde_itens = [item['nQtdItensNFCab'] for item in data]
    #centro = [item['cCentroCab'] for item in data]
    #emissao = [item['dEmissaoNFCab'] for item in data]
    #vencimento = [item['dVencimentoDuplicataCab'] if item['dVencimentoDuplicataCab'] is None else 'N/A' for item in data]
    #fornecedor = [item['cFornecedorCab'] if item['cFornecedorCab'] is None else 'N/A' for item in data]
    #cnpj = [item['cCNPJNFCab'] for item in data]
    #tipo_pc = [item['cTipoPedidoCab'] if item['cTipoPedidoCab'] is None else 'N/A' for item in data]
    #incoterms = [item['cIncotermsCab'] for item in data]
    #peso = [item['nPesoTotalNFCab'] for item in data]
    #valor = [item['nVlrTotalBrutoNFCab'] for item in data]
    #mod_frete = [item['cModalidadeFreteCab'] for item in data]
    #frente = [item['tFrenteCab'] for item in data]
    #cfop = [item['cCFOPItem'] for item in data]
    
    #paths = [item['paths'][0]['path_label'] for item in data]
    #skillLevel = [item['difficulty']['display'] for item in data]
    #links = [base + item['seoslug'] for item in data]

    df=  pd.DataFrame(
      {'cChaveAcessoNFCab': [chave],
       'nQtdItensNFCab': [qtde_itens],
       'cCentroCab': [centro],
       'dEmissaoNFCab': [datetime.datetime.strptime(emissao, '%Y-%m-%d')],
       'dVencimentoDuplicataCab': [datetime.datetime.strptime(vencimento, '%Y-%m-%d')],
       'cFornecedorCab': [fornecedor],
       'cCNPJNFCab': [cnpj],
       'cTipoPedidoCab': [tipo_pc],
       'cIncotermsCab': [incoterms],
       'nPesoTotalNFCab': [peso],
       'nVlrTotalBrutoNFCab': [valor],
       'cModalidadeFreteCab': [mod_frete],
       'tFrenteCab': [frente],
       'cCFOPItem': [cfop]
      })

    #print(df)
    return df

def model_get_input_from_json(data_row):

    # Inicializar dataframe
    df = pd.DataFrame(columns=['cChaveAcessoNFCab', 'nQtdItensNFCab', 'cCentroCab', 'dEmissaoNFCab',
       'dVencimentoDuplicataCab', 'cFornecedorCab', 'cCNPJNFCab', 'cTipoPedidoCab',
       'cIncotermsCab', 'nPesoTotalNFCab', 'nVlrTotalBrutoNFCab', 'cModalidadeFreteCab',
       'tFrenteCab', 'cCFOPItem'])

    # Iterar sobre o dados do json
    for data in data_row["notas"]:
        chave = data['cChaveAcessoNFCab']
        qtde_itens = data['nQtdItensNFCab']
        centro = data['cCentroCab']
        emissao = data['dEmissaoNFCab']
        vencimento = data['dVencimentoDuplicataCab']
        # Ajustar Vencimento
        if pd.isna(vencimento) or vencimento == "":
            if pd.isna(emissao) or emissao == "":
                vencimento = np.NaN
            else:
                vencimento = datetime.datetime.strptime(emissao, '%Y-%m-%d')
        else:
            vencimento = datetime.datetime.strptime(vencimento, '%Y-%m-%d')
        # Ajustar Emissão
        if pd.isna(emissao) or emissao == "":
            emissao = np.NaN
        else:
            emissao = datetime.datetime.strptime(emissao, '%Y-%m-%d')
        fornecedor = data['cFornecedorCab']
        cnpj = data['cCNPJNFCab']
        tipo_pc = data['cTipoPedidoCab']
        incoterms = data['cIncotermsCab']
        peso = data['nPesoTotalNFCab']
        valor = data['nVlrTotalBrutoNFCab']
        mod_frete = data['cModalidadeFreteCab']
        frente = data['tFrenteCab']
        cfop = data['cCFOPItem']
        
        # Criar dataframe auxiliar a partir dos dados
        df_aux =  pd.DataFrame(
            {'cChaveAcessoNFCab': [chave],
            'nQtdItensNFCab': [qtde_itens],
            'cCentroCab': [centro],
            'dEmissaoNFCab': [emissao],
            'dVencimentoDuplicataCab': [vencimento],
            'cFornecedorCab': [fornecedor],
            'cCNPJNFCab': [cnpj],
            'cTipoPedidoCab': [tipo_pc],
            'cIncotermsCab': [incoterms],
            'nPesoTotalNFCab': [peso],
            'nVlrTotalBrutoNFCab': [valor],
            'cModalidadeFreteCab': [mod_frete],
            'tFrenteCab': [frente],
            'cCFOPItem': [cfop]
            })
        
        # Inserir dataframe auxiliar no dataframe principal
        df = pd.concat([df, df_aux], ignore_index=True)
    
    return df

def model_get_input_from_xml(data):
    
    return None

def model_transform_data(df_input):
    df_std = std_input_layer(df_input)
    df_shape = get_df_shape()
    df_trans = transform_layer(df_std, df_shape)
    df_model = model_input_layer(df_trans)
    return df_model, df_std

def model_predict(df_transformed, df_std, model):
    ## Aplicar o modelo ao input
    y_pred = model.predict(df_transformed)
    ## Concatenar resultado
    #df_pred = pd.DataFrame(y_pred,index=df_std.index)
    df_pred = pd.DataFrame(y_pred,index=df_std.index, columns=['pred_value'])
    #print(df_pred)
    ## Aplicar Threshold
    threshold_value = 0.5
    df_pred_bool = df_pred['pred_value'].copy()  
    df_pred_bool[df_pred_bool >= threshold_value] = 1
    df_pred_bool[df_pred_bool < threshold_value] = 0
    df_pred['pred_bool'] = df_pred_bool
    ## Reset index
    df_pred.reset_index(inplace=True)
    return df_pred

def gerar_dict(df_pred):
    dict_pred = { "notas":[]}
    for index, row in df_pred.iterrows():
        nota = {}
        nota["cChaveAcessoNFCab"] = row['cChaveAcessoNFCab']
        nota["pred_value"] = row['pred_value']
        nota["pred_bool"] = row['pred_bool']
        dict_pred["notas"].append(nota)
    return dict_pred

def model_pipeline(data, model, mod):
    # Get input data
    if mod == 'json':
        df_input = model_get_input_from_json(data)
    elif mod == 'json2':
        df_input = model_get_input_from_json2(data)
    elif mod == 'xml':
        df_input = model_get_input_from_xml(data)
    # Tranform input data to fit input model
    df_transformed, df_std = model_transform_data(df_input)
    # Predict from input data
    df_predicted = model_predict(df_transformed, df_std, model)
    # Convert Predictions in Dict
    dict_predicted = gerar_dict(df_predicted)
    # Return df with predictions (ID, predictions)
    return dict_predicted #df_predicted


#def get_input():
    # Help: Obter informações de input e colocar em um dataframe

    # TESTE *** ELIMINAR ***
#    print('Loading excel...')
#    excel_test_path = path/'test'/'zcd_class_test.xlsx'
#    df = pd.read_excel(excel_test_path, converters={'cEmpresaCab':str, 'cCNPJNFCab':str, 'cFornecedorCab':str, 'cSerieNFCab':str, 'cModalidadeFreteCab':str,
#                                         'cItemNFItem':str, 'cPedidoItem':str, 'cItemPCItem':str, 'cUtilizacaoItem':str, 'cSTICMSItem':str,
#                                         'cOrigemMatPCItem':str, 'cOrigemMatXMLItem':str, 'cNCMCTFItem':str})#, sheet_name='Plan2', converters={'cEmpresa':str})
    # *** ELIMINAR ***
#    return df

def get_df_shape():
    cols = ['P01_bVencimentoNull', 'P02_nCondPgto', 'P03_bVlr_3000',
       'P04_bVlr_10000', 'P05_cIncoterms_CIF', 'P05_cIncoterms_FCA',
       'P05_cIncoterms_FOB', 'P05_cIncoterms_SER',
       'P06_cFrente_COPROCESSAMENTO', 'P06_cFrente_INSUMOS', 'P06_cFrente_MRO',
       'P07_bEGX', 'P09_cModalidadeFrete_0', 'P09_cModalidadeFrete_1',
       'P09_cModalidadeFrete_2', 'P09_cModalidadeFrete_3',
       'P09_cModalidadeFrete_4', 'P09_cModalidadeFrete_9', 'P10_nQtdItensNF',
       'P11_nPesoTotalNF', 'P12_bCFOP5']
    df_shape = pd.DataFrame(columns=cols)
    print(df_shape)
    return df_shape

def std_input_layer(df):
    # Help: Tratar camada de entrada (verifica se todos os campos existem, se não há valores faltando, se os tipos
    # estão corretos, etc)

    ## Definir colunas necessárias para o algoritmo
    zcd_model_col = ['cChaveAcessoNFCab', 'dVencimentoDuplicataCab', 'dEmissaoNFCab', 'tFrenteCab', 'cModalidadeFreteCab', 'cCFOPItem',
                    'cIncotermsCab', 'cCentroCab', 'nPesoTotalNFCab', 'nVlrTotalBrutoNFCab', 'nQtdItensNFCab', 
                    'cTipoPedidoCab']
    ## Carregar colunas
    df_zcd = df[zcd_model_col]

    ## Definir index como cChaveAcessoNFCab
    df_zcd.set_index(['cChaveAcessoNFCab'], inplace=True)

    ## Criar coluna target - y
    df_zcd['y_target'] = (df_zcd['cTipoPedidoCab']=='ZCD').astype(float)

    ## Criar coluna vencimento nulo
    df_zcd['P01_bVencimentoNull'] = (pd.isna(df_zcd['dVencimentoDuplicataCab'])).astype(float)

    ## Preencher valores nulos da coluna vencimento com a data de emissão
    df_zcd['dVencimentoDuplicataCab'].fillna(value=df_zcd['dEmissaoNFCab'], inplace=True)

    ## Criar coluna condição de pagamento
    df_zcd['P02_nCondPgto'] = (df_zcd['dVencimentoDuplicataCab']-df_zcd['dEmissaoNFCab'])/np.timedelta64(1,'D')
    #df_zcd['P02_nCondPgto'] = (df_zcd['dVencimentoDuplicataCab']-df_zcd['dEmissaoNFCab']) / 30.0

    ## Preencher valores nulos da coluna P02_nCondPgto com zero
    df_zcd['P02_nCondPgto'].fillna(value=0, inplace=True)

    ## Criar coluna valor total da nota menor que 3000
    df_zcd['P03_bVlr_3000'] = (df_zcd['nVlrTotalBrutoNFCab'] <= 3000).astype(float)

    ## Criar coluna valor total da nota menor que 10000
    df_zcd['P04_bVlr_10000'] = (df_zcd['nVlrTotalBrutoNFCab'] <= 10000).astype(float)

    ## Preencher valores nulos da coluna Incoterms com CIF
    df_zcd['cIncotermsCab'].fillna(value='CIF', inplace=True)

    ## Preencher valores * da coluna Incoterms com CIF
    df_zcd['cIncotermsCab'] = df_zcd['cIncotermsCab'].map(lambda x: 'CIF' if x=='*' else x)

    ## Renomear coluna Incoterm
    df_zcd.rename(columns={'cIncotermsCab': 'P05_cIncoterms'}, inplace=True)

    ## Preencher valores nulos da coluna Frente com VAZIO
    df_zcd['tFrenteCab'].fillna(value='MRO', inplace=True)

    ## Remover erros de escrita das colunas
    df_zcd['tFrenteCab'] = df_zcd['tFrenteCab'].map(lambda x: x[:3] if x.startswith('MRO') else x)
    df_zcd['tFrenteCab'] = df_zcd['tFrenteCab'].map(lambda x: x[:7] if x.startswith('INSUMOS') else x)
    df_zcd['tFrenteCab'] = df_zcd['tFrenteCab'].map(lambda x: x[:15] if x.startswith('COPROCESSAMENTO') else x)

    ## Renomear coluna Frente
    df_zcd.rename(columns={'tFrenteCab': 'P06_cFrente'}, inplace=True)

    ## Criar coluna EGX
    df_zcd['P07_bEGX'] = (df_zcd['cCentroCab'].str.isnumeric() == False).astype(float)

    ## Renomear coluna Centro
    df_zcd.rename(columns={'cCentroCab': 'P08_cCentro'}, inplace=True)

    ## Renomear coluna Modalidade do Frete
    df_zcd.rename(columns={'cModalidadeFreteCab': 'P09_cModalidadeFrete'}, inplace=True)

    ## Renomear coluna Qtde itens da Nota
    df_zcd.rename(columns={'nQtdItensNFCab': 'P10_nQtdItensNF'}, inplace=True)

    ## Renomear coluna Peso Total da Nota
    df_zcd.rename(columns={'nPesoTotalNFCab': 'P11_nPesoTotalNF'}, inplace=True)

    ## Preencher valores nulos da coluna CFOP com VAZIO
    df_zcd['cCFOPItem'].fillna(value='VAZIO', inplace=True)

    ## Se houver /, pegar apenas os 4 primeiros valores
    df_zcd['cCFOPItem'] = df_zcd['cCFOPItem'].map(lambda x: x[:4] if '/' in x else x)

    ## Criar coluna CFOP5
    df_zcd['P12_bCFOP5'] = (df_zcd['cCFOPItem'].str.startswith('5')).astype(float)

    ## Renomear coluna CFOP
    df_zcd.rename(columns={'cCFOPItem': 'P13_cCFOP'}, inplace=True)

    ## Renomear coluna Valor Total da Nota
    df_zcd.rename(columns={'nVlrTotalBrutoNFCab': 'P14_nVlrTotalNF'}, inplace=True)

    ## Dropar colunas que não serão utilizadas
    df_zcd.drop(['dVencimentoDuplicataCab', 'dEmissaoNFCab', 'cTipoPedidoCab', 'P14_nVlrTotalNF', 
                'P13_cCFOP', 'P08_cCentro'], axis=1, inplace=True) # 'P14_nVlrTotalNF',
    
    ## Reordenar colunas em ordem alfabética
    df_zcd.sort_index(axis=1, inplace=True)

    ## Salvar dataframe no resultado
    df_std = df_zcd.copy()
    print(df_std)
    return df_std

def transform_layer(df_std, df_shape):
    # Help: Transformar camada de entrada pré tratada (fazer cálculos de transformação e criar variáveis necessárias)
    
    df_zcd = df_std.copy()

    ## Separar variáveis categóricas de variáveis numéricas 
    catcols = np.array(df_zcd.columns[np.where((df_zcd.dtypes != np.float) & (df_zcd.dtypes != np.int))[0]])
    numcols = np.array(df_zcd.columns[np.where((df_zcd.dtypes == np.float) | (df_zcd.dtypes == np.int))[0]])

    ## Separar dataframes para manipulação
    df_zcd_cat = df_zcd[catcols].copy()
    df_zcd_num = df_zcd[numcols].copy()
    
    ## Obter one hot encoding para as colunas categóricas
    zcd_one_hot = pd.get_dummies(df_zcd_cat).astype(np.float64)

    ## Concatenar resultado
    df_zcd_2 = pd.concat([df_zcd_num, zcd_one_hot], axis=1)

    ## Reordenar colunas em ordem alfabética
    df_zcd_2.sort_index(axis=1, inplace=True)
    print(df_zcd_2)
    ## Definir target
    y_label = 'y_target'
    #y = pd.DataFrame(df_zcd_2[y_label].copy())
    X = df_zcd_2.drop([y_label], axis=1).copy()

    #df_trans = X.copy()
    df_trans = pd.concat([df_shape, X.copy()])
    df_trans = df_trans.reindex(columns=df_shape.columns)
    df_trans.fillna(value=0,inplace=True)
    print(df_trans)
    return df_trans

def model_input_layer(df_trans):
    # Help: Fazer os preparativos necessários para entrar com os parâmetros no modelo

    ## Rescale variables
    z_scaler = joblib.load('model/z_scaler_v01.pkl')
    #z_scaler = StandardScaler()
    #print(df_trans)
    Xz = z_scaler.transform(df_trans)
    print(Xz)
    df_model = Xz.copy()

    return df_model

def predict_layer(df_model, model):
    # Help: Aplicar o modelo aos parâmetros de entrada e obter as predições

    lists=[]
    y_pred = model.predict(df_model)
    for i in range(len(y_pred)):
        if y_pred[i][0]>0.90:   # *** PARÂMETRO ***
            lists.append(1)
        else:
            lists.append(0)
            
    #print(accuracy_score(y_test, lists))

    predictions = lists

    return predictions