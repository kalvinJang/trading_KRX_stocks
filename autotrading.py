import pandas as pd
from datetime import datetime as dt
import datetime
import numpy as np
import requests, json, time, math, ssl, zipfile, os, gc
from pymongo import MongoClient
import urllib.request
import telegram
from telegram.ext import Updater, CommandHandler

def kospi_master_download(base_dir, verbose=False):
    cwd = os.getcwd()
    if (verbose): print(f"current directory is {cwd}")
    ssl._create_default_https_context = ssl._create_unverified_context

    urllib.request.urlretrieve("https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip",
                               base_dir + "/kospi_code.zip")

    os.chdir(base_dir)
    if (verbose): print(f"change directory to {base_dir}")
    kospi_zip = zipfile.ZipFile('kospi_code.zip')
    kospi_zip.extractall()

    kospi_zip.close()

    if os.path.exists("kospi_code.zip"):
        os.remove("kospi_code.zip")
        
def kosdaq_master_download(base_dir, verbose=False):
    cwd = os.getcwd()
    if (verbose): print(f"current directory is {cwd}")
    ssl._create_default_https_context = ssl._create_unverified_context

    urllib.request.urlretrieve("https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip",
                               base_dir + "/kosdaq_code.zip")

    os.chdir(base_dir)
    if (verbose): print(f"change directory to {base_dir}")
    kosdaq_zip = zipfile.ZipFile('kosdaq_code.zip')
    kosdaq_zip.extractall()

    kosdaq_zip.close()

    if os.path.exists("kosdaq_code.zip"):
        os.remove("kosdaq_code.zip")


def get_kospi_master_dataframe(base_dir):
    file_name = base_dir + "/kospi_code.mst"
    tmp_fil1 = base_dir + "/kospi_code_part1.tmp"
    tmp_fil2 = base_dir + "/kospi_code_part2.tmp"

    wf1 = open(tmp_fil1, mode="w")
    wf2 = open(tmp_fil2, mode="w")

    with open(file_name, mode="r", encoding="cp949") as f:
        for row in f:
            rf1 = row[0:len(row) - 228]
            rf1_1 = rf1[0:9].rstrip()
            rf1_2 = rf1[9:21].rstrip()
            rf1_3 = rf1[21:].strip()
            wf1.write(rf1_1 + ',' + rf1_2 + ',' + rf1_3 + '\n')
            rf2 = row[-228:]
            wf2.write(rf2)

    wf1.close()
    wf2.close()

    part1_columns = ['단축코드', '표준코드', '한글명']
    df1 = pd.read_csv(tmp_fil1, header=None, names=part1_columns, engine='python')

    field_specs = [2, 1, 4, 4, 4, #15
                   1, 1, 1, 1, 1, #5
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 9, 5, 5, 1,
                   1, 1, 2, 1, 1,
                   1, 2, 2, 2, 3,
                   1, 3, 12, 12, 8,
                   15, 21, 2, 7, 1,
                   1, 1, 1, 1, 9,
                   9, 9, 5, 9, 8,
                   9, 3, 1, 1, 1
                   ]

    part2_columns = ['그룹코드', '시가총액규모', '지수업종대분류', '지수업종중분류', '지수업종소분류',
                     '제조업', '저유동성', '지배구조지수종목', 'KOSPI200섹터업종', 'KOSPI100',
                     'KOSPI50', 'KRX', 'ETP', 'ELW발행', 'KRX100',
                     'KRX자동차', 'KRX반도체', 'KRX바이오', 'KRX은행', 'SPAC',
                     'KRX에너지화학', 'KRX철강', '단기과열', 'KRX미디어통신', 'KRX건설',
                     'Non1', 'KRX증권', 'KRX선박', 'KRX섹터_보험', 'KRX섹터_운송',
                     'SRI', '기준가', '매매수량단위', '시간외수량단위', '거래정지',
                     '정리매매', '관리종목', '시장경고', '경고예고', '불성실공시',
                     '우회상장', '락구분', '액면변경', '증자구분', '증거금비율',
                     '신용가능', '신용기간', '전일거래량', '액면가', '상장일자',
                     '상장주수', '자본금', '결산월', '공모가', '우선주',
                     '공매도과열', '이상급등', 'KRX300', 'KOSPI', '매출액',
                     '영업이익', '경상이익', '당기순이익', 'ROE', '기준년월',
                     '시가총액', '그룹사코드', '회사신용한도초과', '담보대출가능', '대주가능'
                     ]

    df2 = pd.read_fwf(tmp_fil2, widths=field_specs, names=part2_columns)

    df = pd.merge(df1, df2, how='outer', left_index=True, right_index=True)
    
    # clean temporary file and dataframe
    del (df1)
    del (df2)
    os.remove(tmp_fil1)
    os.remove(tmp_fil2)

    return df

def get_kosdaq_master_dataframe(base_dir):
    file_name = base_dir + "/kosdaq_code.mst"
    tmp_fil1 = base_dir + "/kosdaq_code_part1.tmp"
    tmp_fil2 = base_dir + "/kosdaq_code_part2.tmp"

    wf1 = open(tmp_fil1, mode="w")
    wf2 = open(tmp_fil2, mode="w")
    
    with open(file_name, mode="r", encoding="cp949") as f:
        for row in f:
            rf1 = row[0:len(row) - 228]
            rf1_1 = rf1[0:9].rstrip()
            rf1_2 = rf1[9:21].rstrip()
            rf1_3 = rf1[21:].strip()
            wf1.write(rf1_1 + ',' + rf1_2 + ',' + rf1_3 + '\n')
            rf2 = row[-222:]
            wf2.write(rf2)

    wf1.close()
    wf2.close()

    part1_columns = ['단축코드', '표준코드', '한글명']
    df1 = pd.read_csv(tmp_fil1, header=None, names=part1_columns, engine='python')

    field_specs = [2, 1, 4, 4, 4, #bstp_smal_div_code 지수업종 소분류 코드
                   1, 1, 1, 1, 1, #krx100_issu_yn KRX100 종목 여부
                   1, 1, 1, 1, 1, #etpr_undt_objt_co_yn 기업인수목적회사여부
                   1, 1, 1, 1, 1, #krx_cnst_yn KRX 건설 여부 
                   1, 1, 1, 1, 1, 1, #ksq150_nmix_yn KOSDAQ 150지수여부
                   9, 5, 5, 1, 1, #sltr_yn 정리매매 여부
                   1, 2, 1, 1, 1, #byps_lstn_yn 우회 상장 여부
                   2, 2, 2, 3, 1, #crdt_able 신용주문 가능여부
                   3, 12, 12, 8, 15, #lstn_stcn, 상장주수
                   21, 2, 7, 1,1, #ssts_hot_yn 공매도과열종목여부
                   1, 1, 9, 9, 9, #op_prfi 경상이익
                   5, 9, 8, 9, 3, #grp_code 그룹사 코드
                   1, 1, 1
                   ]

    part2_columns = ['그룹코드', '시가총액규모', '지수업종대분류', '지수업종중분류', '지수업종소분류',
                     '벤처기업 여부', '저유동성', 'KRX', 'ETP 상품구분코드', 'KRX100',
                     'KRX자동차', 'KRX반도체', 'KRX바이오', 'KRX은행', 'SPAC',
                     'KRX에너지화학', 'KRX철강', '단기과열', 'KRX미디어통신', 'KRX건설',
                     'Non1', 'KRX증권', 'KRX선박', 'KRX섹터_보험', 'KRX섹터_운송',
                     'KOSDAQ150지수여부', '기준가', '매매수량단위', '시간외수량단위', '거래정지',
                     '정리매매', '관리종목', '시장경고', '경고예고', '불성실공시',
                     '우회상장', '락구분', '액면가변경', '증자구분', '증거금비율',
                     '신용가능', '신용기간', '전일거래량', '액면가', '상장일자',
                     '상장주수', '자본금', '결산월', '공모가', '우선주',
                     '공매도과열', '이상급등', 'KRX300','매출액',
                     '영업이익', '경상이익', '당기순이익', 'ROE', '기준년월',
                     '시가총액', '그룹사코드', '회사신용한도초과', '담보대출가능', '대주가능'
                     ]

    df2 = pd.read_fwf(tmp_fil2, widths=field_specs, names=part2_columns)

    df = pd.merge(df1, df2, how='outer', left_index=True, right_index=True)
    
    # clean temporary file and dataframe
    del (df1)
    del (df2)
    os.remove(tmp_fil1)
    os.remove(tmp_fil2)

    return df


def drawdown(cum):
    cumulative = cum+1
    highwatermark = cumulative.cummax()
    drawdown = (cumulative / highwatermark) -1
    return drawdown
def max_dd(cum):
    return np.min(drawdown(cum))

def cagr(cash_hist):
    return (cash_hist[-1]/cash_hist[0])**(252/len(cash_hist))-1

def stdv(returns):
    return np.std(returns)*math.sqrt(252)

def sharpe(returns):
    return cagr(returns.cumprod())/stdv(returns)

def winsorize(x):
    if x==0:
        return 1
    elif (x>0.69) and (x<1.31):
        return x
    else:
        return 1

class strategy(object):
    def __init__(self, close, volume, acml, low, high, open, returns, k, t, s):
        self.returns = returns
        self.close = close
        self.open = open
        self.high = high
        self.low = low
        self.trade_vol = acml
        self.trade_amount = volume
        self.vwap = self.trade_amount / self.trade_vol
        self.adv20 = self.trade_amount.rolling(20).mean()
        self.isEW = True
        self.isascending=True
        self.isupperratio=0
        self.how_max = 0.1
        self.key = k
        self.token = t
        self.secret = s
        
    def cal_weight(self, formula, isEW=True, isascending=True, isupperratio=0, how_max=1):

        mask = ((self.trade_amount.rolling(5, min_periods=1).min()>5*(10**7))*1).replace(0, np.nan)

        if isEW==True:
            formula *= mask
            weight = formula.apply(lambda x: x/x.sum(),axis=1).replace(np.nan, 0)
            weight = weight.applymap(lambda x: how_max if x>how_max else x)
            self.cash = 1- weight.sum(1)
        elif isEW==False:
            ratio = formula.apply(lambda x: x.rank(ascending=isascending)/(x.rank(ascending=isascending).max()),axis=1)
            formula2 = (ratio>=isupperratio)*1
            formula2 *= mask
            formula3 = formula2.replace(0, np.NaN)
            formula4 = formula * formula3
            weight = formula4.apply(lambda x: x.rank(ascending=isascending)/(x.rank(ascending=isascending).sum()), axis=1)
        self.weight = weight.replace(np.nan, 0)
        self.last_day_weight = weight.iloc[-1,:]
        
    def select_sell_or_buy(self, buy_list):
        '''잔고조회'''
        url = 'https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/trading/inquire-balance'
        headers = {
            'content-type': 'application/json',
            'authorization': self.token,
            'appkey': self.key,
            'appsecret': self.secret,
            'tr_id': 'TTTC8434R' 
        }
        body = {
            "CANO": "63576598",
            "ACNT_PRDT_CD": '01',
            "AFHR_FLPR_YN":  'N',
            "OFL_YN": "N",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        res = requests.get(url, headers=headers, params=body)
        holding = pd.DataFrame(json.loads(res.text)['output1']).loc[:,['pdno','prpr', 'hldg_qty']]
        holding['prpr']=holding['prpr'].astype(int)
        holding['hldg_qty']=holding['hldg_qty'].astype(int)
        hold_stocks = []
        for i in range(holding.shape[0]):
            hold_stocks.append(holding.iloc[i].tolist())
        hold_stocks
        
        order_stocks = buy_list
        order_weight = 1/len(buy_list)
        order_amount = (order_weight*1000000/self.close.iloc[-1, :][order_stocks]).astype(int)
        order_price = self.close.loc[:,order_stocks].iloc[-1]
        tomorrow_stocks = [x for x in zip(order_stocks, order_price.astype(int), order_amount)]
        temp= pd.DataFrame(tomorrow_stocks).groupby([0,1]).sum().reset_index()
        tomorrow_stocks = temp.values.tolist()
        order_temp = pd.concat([pd.DataFrame(hold_stocks).set_index(0),pd.DataFrame(tomorrow_stocks).set_index(0).applymap(lambda x: 0)]) - pd.concat([pd.DataFrame(tomorrow_stocks).set_index(0), pd.DataFrame(hold_stocks).set_index(0).applymap(lambda x: 0)])
        order_control = order_temp.groupby([0]).sum()
        self.sell_stocks = [x for x in zip(order_control[order_control[2]>0].index, self.close.loc[:,order_control[order_control[2]>0].index].iloc[-1].astype(int), order_control[order_control[2]>0][2].astype(int).astype(str))]
        self.buy_stocks = [x for x in zip(order_control[order_control[2]<0].index, self.close.loc[:,order_control[order_control[2]<0].index].iloc[-1].astype(int), (-1 *order_control[order_control[2]<0][2]).astype(int).astype(str))]
            
    def st1(self):
        temp = self.close.rolling(7).mean()-self.close
        temp2 = temp.apply(lambda x: x/np.abs(x).sum())
        temp3 = self.vwap.rolling(230).corr(self.close.shift(5))
        formula = temp2 + temp3.apply(lambda x: 20*x/np.abs(x).sum())
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio, self.how_max)
        buy = [x for x in pd.DataFrame(self.last_day_weight).sort_values(by=self.last_day_weight.name, ascending=False).replace(0, np.nan).dropna().head(8).index]
        return buy
        
    def st2(self):
        cond1 = self.open/(self.close.shift(2)) > (self.open/(self.close.shift(2))).rolling(60).std()
        x = (0.3* self.low * self.trade_vol * 0.7 * self.low)
        cond2 = x**(-0.3) > (x**(-0.3)).rolling(60).std()
        cond3 = (self.low - self.low.shift(2)) > (0.3 * self.high/self.low * 0.7 * self.high)
        temp = cond1*cond2*cond3*1
        formula = temp.replace(0, np.nan)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio, self.how_max)
        buy = [x for x in pd.DataFrame(self.last_day_weight).sort_values(by=self.last_day_weight.name, ascending=False).replace(0, np.nan).dropna().head(8).index]
        return buy

        
def order(key, token, secret, order_type='sell', order_list=[], after_four=False):
    bot = telegram.Bot(token='*********************************')
    if order_type=='buy':
        for i in range(len(order_list)):
            code = order_list[i][0]
            qty = order_list[i][2]
            unpr = str(order_list[i][1])
            headers = {
              'content-type': 'application/json',
              'authorization': token,
              'appkey': key,
              'appsecret': secret,
              'tr_id': 'TTTC0802U' 
            }
            if after_four==False:
                body = {
                    "CANO": "63576598",
                    "ACNT_PRDT_CD": "01",
                    "PDNO": code,
                    "ORD_DVSN": "06",
                    "ORD_QTY": qty,
                    "ORD_UNPR": '0', 
                }
            elif after_four==True:
                body = {
                    "CANO": "63576598",
                    "ACNT_PRDT_CD": "01",
                    "PDNO": code,
                    "ORD_DVSN": "07", 
                    "ORD_QTY": qty,
                    "ORD_UNPR": unpr, 
                }
            elif after_four=='0':
                body = {
                    "CANO": "63576598",
                    "ACNT_PRDT_CD": "01", 
                    "PDNO": code,
                    "ORD_DVSN": "01",
                    "ORD_QTY": qty,
                    "ORD_UNPR": '0', 
                }
            url = 'https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/trading/order-cash'
            res = requests.post(url, data=json.dumps(body), headers=headers)
            for i in ['*******', '*********']:
                bot.sendMessage(chat_id=i, text='BUY order')
                bot.sendMessage(chat_id=i, text=res.text)


    elif order_type=='sell':
        for i in range(len(order_list)):
            code = order_list[i][0]
            qty = order_list[i][2]
            unpr = str(order_list[i][1]) 
            headers = {
              'content-type': 'application/json',
              'authorization': token,
              'appkey': key,
              'appsecret': secret,
              'tr_id': 'TTTC0801U' 
            }
            if after_four==False:
                body = {
                    "CANO": "63576598",
                    "ACNT_PRDT_CD": "01",
                    "PDNO": code,
                    "ORD_DVSN": "06",
                    "ORD_QTY": qty,
                    "ORD_UNPR": '0',  
                }
            elif after_four==True:
                body = {
                    "CANO": "63576598",
                    "ACNT_PRDT_CD": "01",
                    "PDNO": code,
                    "ORD_DVSN": "07", 
                    "ORD_QTY": qty,
                    "ORD_UNPR": unpr, 
                }
            elif after_four=='0':
                body = {
                    "CANO": "63576598",
                    "ACNT_PRDT_CD": "01",
                    "PDNO": code,
                    "ORD_DVSN": "01",
                    "ORD_QTY": qty,
                    "ORD_UNPR": '0', 
                }
            url = 'https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/trading/order-cash'
            res = requests.post(url, data=json.dumps(body), headers=headers)
            for i in ['*******', '*********']:
                bot.sendMessage(chat_id=i, text='SELL order')
                bot.sendMessage(chat_id=i, text=res.text) 