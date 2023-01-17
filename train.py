import numpy as np
from Stock import Stock

from returnInputs import returnInput
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def returnDailyLabels(a,b):
    inputs = returnInput(a, b)
    results = inputs[1:,0].reshape(-1,1).astype(float)
    return results

SA = Stock('005930');   sa,_=SA.returnPrices();     sadata=SA.returnIdx();
SAC = Stock('029780');  sac,_=SAC.returnPrices();   sacdata=SAC.returnIdx();
SAL = Stock('032830');  sal,_=SAL.returnPrices();   saldata=SAL.returnIdx();
SAS = Stock('006400');  sas,_=SAS.returnPrices();   sasdata=SAS.returnIdx();
HDC = Stock('005380');  hdc,_=HDC.returnPrices();   hdcdata=HDC.returnIdx();
HDM = Stock('012330');  hdm,_=HDM.returnPrices();   hdmdata=HDM.returnIdx();
HDS = Stock('004020');  hds,_=HDS.returnPrices();   hdsdata=HDS.returnIdx();
SK = Stock('000660');   sk,_=SK.returnPrices();     skdata=SK.returnIdx();
SKI = Stock('096770');  ski,_=SKI.returnPrices();   skidata=SKI.returnIdx();
SKC = Stock('011790');  skc,_=SKC.returnPrices();   skcdata=SKC.returnIdx();
SKG = Stock('018670');  skg,_=SKG.returnPrices();   skgdata=SKG.returnIdx();
SKT = Stock('000660');  skt,_=SKT.returnPrices();   sktdata=SKT.returnIdx();
SKN = Stock('001740');  skn,_=SKN.returnPrices();   skndata=SKN.returnIdx();
DAU = Stock('023590');  dau,_=DAU.returnPrices();   daudata=DAU.returnIdx();
DW = Stock('049770');   dw,_=DW.returnPrices();     dwdata=DW.returnIdx();
DWP=Stock('003090');    dwp,_=DWP.returnPrices();   dwpdata=DWP.returnIdx();
SFA = Stock('056190');  sfa,_=SFA.returnPrices();   sfadata=SFA.returnIdx();
GB = Stock('024110');   gb,_=GB.returnPrices();     gbdata=GB.returnIdx();
LGU = Stock('032640');  lgu,_=LGU.returnPrices();   lgudata=LGU.returnIdx();
LGEL = Stock('005380'); lgel,_=LGEL.returnPrices(); lgeldata=LGEL.returnIdx();
NAV = Stock('035420');  nav,_=NAV.returnPrices();   navdata=NAV.returnIdx();
NCS = Stock('036570');  ncs,_=NCS.returnPrices();   ncsdata=NCS.returnIdx();
CEL = Stock('068270');  cel,_=CEL.returnPrices();   celdata=CEL.returnIdx();
SOIL = Stock('010950'); soil,_=SOIL.returnPrices(); soildata=SOIL.returnIdx();
ORIH = Stock('001800'); orih,_=ORIH.returnPrices(); orihdata=ORIH.returnIdx();
GANG = Stock('035250'); gang,_=GANG.returnPrices(); gangdata=GANG.returnIdx();
KIA = Stock('000270');  kia,_=KIA.returnPrices();   kiadata=KIA.returnIdx();
KA = Stock('035720');   ka,_=KA.returnPrices();     kadata=KA.returnIdx();
KT = Stock('030200');   kt,_=KT.returnPrices();     ktdata=KT.returnIdx();
KOA = Stock('003490');  koa,_=KOA.returnPrices();   koadata=KOA.returnIdx();
KORA = Stock('010130'); kora,_=KORA.returnPrices(); koradata=KORA.returnIdx();
HANS = Stock('009830'); hans,_=HANS.returnPrices(); hansdata=HANS.returnIdx();
POC = Stock('003670');  poc,_=POC.returnPrices();   pocdata=POC.returnIdx();
POH = Stock('005490');  poh,_=POH.returnPrices();   pohdata=POH.returnIdx();
NHN = Stock('181710');  nhn,_=NHN.returnPrices();   nhndata=NHN.returnIdx();
KL= Stock('003690');    kl,_=KL.returnPrices();     kldata=KL.returnIdx();
UC=Stock('008730');     uc,_=UC.returnPrices();     ucdata=UC.returnIdx();
LL=Stock('032350');     ll,_=LL.returnPrices();     lldata=LL.returnIdx();
ISD=Stock('010780');    isd,_=ISD.returnPrices();   isddata=ISD.returnIdx();
DT=Stock('145720');     dt,_=DT.returnPrices();     dtdata=DT.returnIdx();
ASI=Stock('020560');    asi,_=ASI.returnPrices();   asidata=ASI.returnIdx();
DB=Stock('012510');     db,_=DB.returnPrices();     dbdata=DB.returnIdx();
SP=Stock('019170');     sp,_=SP.returnPrices();     spdata=SP.returnIdx();
HS=Stock('093370');     hs,_=HS.returnPrices();     hsdata=HS.returnIdx();
HSAM=Stock('009240');   hsam,_=HSAM.returnPrices(); hsamdata=HSAM.returnIdx();
PUNG = Stock('103140'); pung,_=PUNG.returnPrices(); pungdata=PUNG.returnIdx();
COLM= Stock('161890');  colm,_=COLM.returnPrices(); colmdata=COLM.returnIdx();
DANG= Stock('185750');  dang,_=DANG.returnPrices(); dangdata=DANG.returnIdx();
HANA= Stock('039130');  hana,_=HANA.returnPrices(); hanadata=HANA.returnIdx();
JJ=Stock('089590');     jj,_=JJ.returnPrices();     jjdata=JJ.returnIdx();
#LJ=Stock('280360');     lj,_=LJ.returnPrices();     ljdata=LJ.returnIdx();
#ORI = Stock('271560');  ori,_=ORI.returnPrices();   oridata=ORI.returnIdx();
#LGE = Stock('373220');  lge,_=LGE.returnPrices();   lgedata=LGE.returnIdx();

train_inputs = []
train_labels = []
train_inputs.append(returnInput(sa, sadata))
train_inputs.append(returnInput(sac, sacdata))
train_inputs.append(returnInput(sal, saldata))
train_inputs.append(returnInput(sas, sasdata))
train_inputs.append(returnInput(hdc, hdcdata))
train_inputs.append(returnInput(hdm, hdmdata))
train_inputs.append(returnInput(hds, hdsdata))
train_inputs.append(returnInput(sk, skdata))
#train_inputs.append(returnInput(ski, skidata))
train_inputs.append(returnInput(skc, skcdata))
train_inputs.append(returnInput(skg, skgdata))
train_inputs.append(returnInput(skt, sktdata))
train_inputs.append(returnInput(skn, skndata))
#train_inputs.append(returnInput(dau, daudata))
train_inputs.append(returnInput(dw, dwdata))
#train_inputs.append(returnInput(dwp, dwpdata))
train_inputs.append(returnInput(sfa, sfadata))
train_inputs.append(returnInput(gb, gbdata))
train_inputs.append(returnInput(lgu, lgudata))
train_inputs.append(returnInput(lgel, lgeldata))
train_inputs.append(returnInput(nav, navdata))
train_inputs.append(returnInput(ncs, ncsdata))
train_inputs.append(returnInput(cel, celdata))
train_inputs.append(returnInput(soil, soildata))
#train_inputs.append(returnInput(orih, orihdata))
#train_inputs.append(returnInput(gang, gangdata))
train_inputs.append(returnInput(kia, kiadata))
train_inputs.append(returnInput(ka, kadata))
train_inputs.append(returnInput(kt, ktdata))
#train_inputs.append(returnInput(koa, koadata))
train_inputs.append(returnInput(kora, koradata))
train_inputs.append(returnInput(hans, hansdata))
train_inputs.append(returnInput(poc, pocdata))
train_inputs.append(returnInput(poh, pohdata))
train_inputs.append(returnInput(nhn, nhndata))
#train_inputs.append(returnInput(kl, kldata))
#train_inputs.append(returnInput(uc, ucdata))
#train_inputs.append(returnInput(ll, lldata))
train_inputs.append(returnInput(isd, isddata))
train_inputs.append(returnInput(dt, dtdata))
#train_inputs.append(returnInput(asi, asidata))
train_inputs.append(returnInput(db, dbdata))
#train_inputs.append(returnInput(sp, spdata))
train_inputs.append(returnInput(hs, hsdata))
train_inputs.append(returnInput(hsam, hsamdata))
train_inputs.append(returnInput(pung, pungdata))
train_inputs.append(returnInput(colm, colmdata))
train_inputs.append(returnInput(dang, dangdata))
#train_inputs.append(returnInput(hana, hanadata))
#train_inputs.append(returnInput(jj, jjdata))
train_inputs = np.array(train_inputs)

train_labels.append(returnDailyLabels(sa, sadata))
train_labels.append(returnDailyLabels(sac, sacdata))
train_labels.append(returnDailyLabels(sal, saldata))
train_labels.append(returnDailyLabels(sas, sasdata))
train_labels.append(returnDailyLabels(hdc, hdcdata))
train_labels.append(returnDailyLabels(hdm, hdmdata))
train_labels.append(returnDailyLabels(hds, hdsdata))
train_labels.append(returnDailyLabels(sk, skdata))
#train_labels.append(returnDailyLabels(ski, skidata))
train_labels.append(returnDailyLabels(skc, skcdata))
train_labels.append(returnDailyLabels(skg, skgdata))
train_labels.append(returnDailyLabels(skt, sktdata))
train_labels.append(returnDailyLabels(skn, skndata))
#train_labels.append(returnDailyLabels(dau, daudata))
train_labels.append(returnDailyLabels(dw, dwdata))
#train_labels.append(returnDailyLabels(dwp, dwpdata))
train_labels.append(returnDailyLabels(sfa, sfadata))
train_labels.append(returnDailyLabels(gb, gbdata))
train_labels.append(returnDailyLabels(lgu, lgudata))
train_labels.append(returnDailyLabels(lgel, lgeldata))
train_labels.append(returnDailyLabels(nav, navdata))
train_labels.append(returnDailyLabels(ncs, ncsdata))
train_labels.append(returnDailyLabels(cel, celdata))
train_labels.append(returnDailyLabels(soil, soildata))
#train_labels.append(returnDailyLabels(orih, orihdata))
#train_labels.append(returnDailyLabels(gang, gangdata))
train_labels.append(returnDailyLabels(kia, kiadata))
train_labels.append(returnDailyLabels(ka, kadata))
train_labels.append(returnDailyLabels(kt, ktdata))
#train_labels.append(returnDailyLabels(koa, koadata))
train_labels.append(returnDailyLabels(kora, koradata))
train_labels.append(returnDailyLabels(hans, hansdata))
train_labels.append(returnDailyLabels(poc, pocdata))
train_labels.append(returnDailyLabels(poh, pohdata))
train_labels.append(returnDailyLabels(nhn, nhndata))
#train_labels.append(returnDailyLabels(kl, kldata))
#train_labels.append(returnDailyLabels(uc, ucdata))
#train_labels.append(returnDailyLabels(ll, lldata))
train_labels.append(returnDailyLabels(isd, isddata))
train_labels.append(returnDailyLabels(dt, dtdata))
#train_labels.append(returnDailyLabels(asi, asidata))
train_labels.append(returnDailyLabels(db, dbdata))
#train_labels.append(returnDailyLabels(sp, spdata))
train_labels.append(returnDailyLabels(hs, hsdata))
train_labels.append(returnDailyLabels(hsam, hsamdata))
train_labels.append(returnDailyLabels(pung, pungdata))
train_labels.append(returnDailyLabels(colm, colmdata))
train_labels.append(returnDailyLabels(dang, dangdata))
#train_labels.append(returnDailyLabels(hana, hanadata))
#train_labels.append(returnDailyLabels(jj, jjdata))
train_labels = np.array(train_labels)[:,-1,:]


dmodel = Sequential([
    layers.LSTM(256, input_shape=(train_inputs.shape[1],train_inputs.shape[2]), return_sequences=True),
    layers.Dense(1, activation = 'relu')
])

dmodel.compile(loss='mse',optimizer='adam', metrics='mae')
dmodel.summary()
dmodel.fit(train_inputs, train_labels,
    epochs=200, batch_size=1
)
dmodel.save('dailymodel.h5')