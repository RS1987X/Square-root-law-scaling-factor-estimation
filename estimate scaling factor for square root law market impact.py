# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 19:27:13 2022

@author: richa
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math
from datetime import date
import statsmodels.api as sm


tday = date.today()
tday_str = tday.strftime("%Y-%m-%d")


tickers = ["AZN.ST","ATCO-A.ST","ABB.ST","INVE-B.ST","VOLV-B.ST","NDA-SE.ST","EQT.ST","ERIC-B.ST","HEXA-B.ST","SAND.ST","NOKIA-SEK.ST","ASSA-B.ST","HM-B.ST","EVO.ST", \
            "SEB-A.ST","EPI-A.ST","VOLCAR-B.ST","SWED-A.ST","SHB-A.ST","ESSITY-B.ST","LATO-B.ST","NIBE-B.ST","TELIA.ST","STE-R.ST","ALFA.ST","INDU-C.ST","LUND-B.ST","8TRA.ST", \
            "SWMA.ST","SCA-B.ST","BALD-B.ST","BOL.ST","LIFCO-B.ST","LUNE.ST","EMBRAC-B.ST","SKF-B.ST","SKA-B.ST","TEL2-B.ST","GETI-B.ST","SAGA-B.ST","SBB-B.ST","INDT.ST", \
            "ALIV-SDB.ST","KINV-B.ST","STOR-B.ST","CAST.ST","HOLM-B.ST","HUSQ-B.ST","SINCH.ST","TREL-B.ST","BEIJ-B.ST","LUMI.ST","SSAB-A.ST","ELUX-B.ST", \
            "SOBI.ST","VITR.ST","AXFO.ST","WALL-B.ST","AAK.ST","THULE.ST","SWEC-B.ST","ADDT-B.ST","AZA.ST","FABG.ST","MCOV-B.ST","SECU-B.ST","EKTA-B.ST","HPOL-B.ST",\
            "SAVE.ST","VNE-SDB.ST","CORE-B.ST","ALIF-B.ST","DOM.ST","TIETOS.ST","PEAB-B.ST","SAAB-B.ST","BILL.ST","INTRUM.ST",\
            "NENT-B.ST","WIHL.ST","TRUE-B.ST","SECT-B.ST","NYF.ST","CINT.ST","HUFV-A.ST","VIMIAN.ST","NOLA-B.ST","ATRLJ-B.ST","ARJO-B.ST","TIGO-SDB.ST","JM.ST","PNDX-B.ST","AFRY.ST",\
            "MIPS.ST","KIND-SDB.ST","BURE.ST","BRAV.ST","CATE.ST","FPAR-A.ST","HMS.ST","LIAB.ST","TROAX.ST","ARION-SDB.ST","PDX.ST","SF.ST","LOOMIS.ST","SYSR.ST","MYCR.ST",\
            "EPRO-B.ST","INSTAL.ST","NCC-B.ST","LUG.ST","CRED-A.ST","RATO-B.ST","VOLO.ST","HEM.ST","NP3.ST","FOI-B.ST","HTRO.ST","DIOS.ST","SDIP-B.ST","BEIA-B.ST",\
            "PLAZ-B.ST","KFAST-B.ST","VIT-B.ST","OX2.ST","VESTUM.ST","BICO.ST","BUFAB.ST","BILI-A.ST","KARO.ST","FIL.ST","NCAB.ST","ANOD-B.ST","BIOT.ST",\
            "MTRS.ST","SECARE.ST","CARY.ST","HEBA-B.ST","ALM.ST","BFG.ST","BOOZT.ST","CIBUS.ST","SKIS-B.ST","RVRC.ST","SUS.ST","DUST.ST","ALLIGO-B.ST","BHG.ST","GRNG.ST",\
            "SAS.ST","MTG-B.ST","STORY-B.ST","VNV.ST"]
#tickers = ["AZN.ST", "ATCO-A.ST"]
market_impact_save = []
returns = []

for x in tickers:

    hist = yf.download(x, start='2020-01-01', end='2022-01-01')

    
    close_prices = hist["Adj Close"]#.dropna(how='all').fillna(0)
    #high_prices = hist["High"]
    #low_prices = hist["Low"]
    #avg_prices = (close_prices + high_prices + low_prices)/3
    #volume traded each day
    Q = hist["Volume"].dropna(how='all').fillna(0).astype(float)
    #average daily volume traded each day
    V = Q.rolling(60).mean().shift(1)
    #calculate daily returns
    ret_daily = close_prices.pct_change()
    #calculate volatility
    sigma = ret_daily.rolling(60).std().shift(1)
    
    neg_ind = ret_daily < 0
    pos_ind = ret_daily >= 0
        
    market_impact = -1*neg_ind*sigma*np.sqrt(Q/V) + pos_ind*sigma*np.sqrt(Q/V)
    #market_impact = sigma*np.sqrt(Q/V)
    
    
    market_impact_save.append(market_impact.values)
    returns.append(ret_daily.values)

market_impact_df = pd.DataFrame(market_impact_save)
market_impact_df = market_impact_df.melt()["value"]

returns_df = pd.DataFrame(returns)
returns_df = returns_df.melt()["value"]
#market_impact_df = pd.DataFrame(list(market_impact_save.items()),columns=["index", "returns"])
#market_impact_df = market_impact_df.set_index("index")
#market_impact_df = market_impact_df.sort_index()


#returns_df = pd.DataFrame(list(returns.items()),columns=["index", "returns"])
#returns_df = returns_df.set_index("index")
#returns_df = returns_df.sort_index()

plt.scatter(market_impact_df,returns_df)

#x = sm.add_constant(x1) # adding a constant
lm = sm.OLS(returns_df,market_impact_df,missing='drop').fit() # fitting the model
#lm.predict(x)
print(lm.summary())